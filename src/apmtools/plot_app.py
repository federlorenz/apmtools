"""Standalone, replaceable-dataset Shiny plotting application.

Pickle loading is intentionally restricted to explicitly selected local files.
Only open trusted pickle files: Python pickle can execute arbitrary code.
"""
from __future__ import annotations

from pathlib import Path
import pickle
import threading


def load_summary_pickle(path):
    """Load a trusted pickle and verify its concrete type before use."""
    from .classes import Summary
    path = Path(path)
    if path.suffix.lower() != ".pkl":
        raise ValueError("Select a .pkl file containing an apmtools Summary")
    # IMPORTANT: pickle executes code during deserialization. The type check
    # cannot protect against untrusted files; the user must trust the source.
    with path.open("rb") as handle:
        data = pickle.load(handle)
    if not isinstance(data, Summary):
        raise TypeError("The selected pickle does not contain an apmtools Summary")
    if data.empty:
        raise ValueError("The selected Summary is empty")
    # Older pickles may lack attributes not registered in Summary._metadata.
    # pandas.copy() invokes Summary.__finalize__ and requires these fields.
    if not hasattr(data, "variable"):
        data.variable = None
    if not hasattr(data, "metadata"):
        data.metadata = set()
    if not hasattr(data, "description"):
        data.description = ""
    return data


class PlotApp:
    """Launch a Shiny plot workspace with an optional initial Summary.

    Examples
    --------
    >>> PlotApp(summary).run()
    >>> PlotApp().run()
    """

    def __init__(self, summary=None, *, x=None, y=None, group_by=None,
                 plot_type="line", title="Summary plot", backend="plotly"):
        from .classes import Summary
        if summary is not None and not isinstance(summary, Summary):
            raise TypeError("PlotApp expects an apmtools Summary or None")
        self.original_summary = None
        self.summary = None
        if summary is not None:
            self._set_dataset(summary)
        self.options = dict(x=x, y=y, group_by=group_by,
                            plot_type=plot_type, title=title, backend=backend)

    def _set_dataset(self, summary):
        """Retain the source and make a lightweight working DataFrame.

        The source is retained for future explicit metadata persistence. The
        working frame shares storage until pandas Copy-on-Write separates
        modified columns. Mutable Python objects inside cells or metadata
        remain shared and must not be mutated in place.
        """
        from .classes import Summary
        if not isinstance(summary, Summary):
            raise TypeError("PlotApp expects an apmtools Summary")
        self.original_summary = summary
        # Do not change process-wide pandas settings here.
        # With older pandas and CoW disabled, a shallow copy may share writes.
        self.summary = summary.copy(deep=False)
        # Explicitly retained presets belong to the source Summary. Avoid
        # mutating a shared list accidentally through the working frame.
        self.summary.plots = None if getattr(summary, "plots", None) is None else list(summary.plots)

    def create_app(self):
        """Build an app whose complete plotting workspace is replaced on load."""
        from shiny import App, module, reactive, render, ui
        from .shiny.ui import _build_summary_ui_server
        from .plot_configs import PlotConfigManager
        import shinyswatch

        app_ui = ui.page_fluid(
            ui.tags.button("Hide toolbars", type="button", class_="btn btn-sm btn-outline-secondary my-0", id="standalone-toolbar-collapse-toggle", **{"aria-expanded": "true"}),
            ui.div(
                ui.input_action_button("open_dataset", "Open Dataset", class_="btn-outline-primary"),
                ui.div(id="standalone-save-dataset-slot"),
                ui.output_text("dataset_status", inline=True),
                ui.div(shinyswatch.theme_picker_ui(), class_="summary-app-theme ms-auto"),
                class_="d-flex align-items-center gap-1 my-0 standalone-dataset-toolbar",
            ),
            ui.tags.script("""
              (() => {
                const move = () => {
                  const slot = document.getElementById('standalone-save-dataset-slot');
                  const button = document.querySelector('#summary-dataset-toolbar button');
                  if (slot && button && button.parentElement !== slot) slot.appendChild(button);
                };
                new MutationObserver(move).observe(document.body, {childList:true, subtree:true});
                document.addEventListener('DOMContentLoaded', move);
                move();
              })();
            """),
            ui.output_ui("dataset_workspace"),
            title="PlotApp",
        )

        def server(input, output, session):
            dataset = reactive.value(self.summary)
            generation = reactive.value(0)
            status = reactive.value("" if self.summary is not None else "No dataset loaded")
            # Keep a fresh module per dataset, ensuring no old reactive
            # inputs, plot tabs, or legend editors leak into the new dataset.
            mounted = set()
            ui_factories = {}
            mounted_generation = reactive.value(-1)
            # Theme picker must be bound at the outer Shiny session, not
            # inside a dynamically mounted workspace module.
            shinyswatch.theme_picker_server()

            @render.text
            def dataset_status():
                return status.get()

            @reactive.effect
            @reactive.event(input.open_dataset)
            def show_dataset_dialog():
                ui.modal_show(ui.modal(
                    ui.p("Choose a trusted .pkl file containing a pickled apmtools Summary."),
                    ui.input_file("summary_pickle", "Select Summary pickle", accept=[".pkl"], multiple=False),
                    ui.p("Warning: Python pickle files can execute code. Open only files you trust."),
                    title="Open Dataset", easy_close=True,
                    footer=ui.modal_button("Cancel"),
                ))

            @reactive.effect
            @reactive.event(input.summary_pickle)
            def receive_dataset():
                files = input.summary_pickle()
                if not files:
                    return
                try:
                    # Shiny places the browser-selected upload in a temporary
                    # server-side file. The browser file picker navigates local
                    # folders and sends the selected file to the running app.
                    file = files[0]
                    if Path(file["name"]).suffix.lower() != ".pkl":
                        raise ValueError("Only .pkl files are supported")
                    replacement = load_summary_pickle(file["datapath"])
                except Exception as exc:
                    status.set(f"Could not load dataset: {exc}")
                    ui.notification_show(f"Could not load dataset: {exc}", type="error", duration=8)
                    return
                self._set_dataset(replacement)
                dataset.set(self.summary)
                generation.set(generation.get() + 1)
                status.set(f"Loaded: {file['name']}")
                ui.modal_remove()

            def register_workspace(data, token):
                """Register each module server once, outside a reactive render."""
                name = f"dataset_{token}"
                if name in mounted:
                    return
                # Shiny module UI IDs are namespaced at UI construction time.
                # Building inner_ui before entering @module.ui leaves toolbar
                # inputs unprefixed, while the module server listens for
                # dataset_N-add_tab / dataset_N-add_table.
                manager = PlotConfigManager(self.original_summary)
                @module.ui
                def workspace_ui():
                    inner_ui, _ = _build_summary_ui_server(
                        data, **self.options, include_theme_picker=False, config_manager=manager)
                    return inner_ui

                @module.server
                def workspace_server(input, output, session):
                    _, inner_server = _build_summary_ui_server(
                        data, **self.options, include_theme_picker=False, config_manager=manager)
                    return inner_server(input, output, session)

                ui_factories[name] = workspace_ui
                mounted.add(name)
                workspace_server(name)
                mounted_generation.set(token)

            # The initial workspace MUST be registered in the normal server
            # initialization context. Mounting it from a reactive effect
            # leaves its nested event handlers inactive in some Shiny versions.
            if self.summary is not None:
                register_workspace(self.summary, 0)

            @reactive.effect
            @reactive.event(generation)
            def mount_replacement_workspace():
                token = generation.get()
                if token == 0:
                    return
                data = dataset.get()
                if data is not None:
                    register_workspace(data, token)

            @render.ui
            def dataset_workspace():
                data = dataset.get()
                if data is None:
                    return ui.div(ui.h3("No dataset loaded"),
                                  ui.p("Use Open Dataset to select a trusted Summary .pkl file."),
                                  class_="p-4")
                name = f"dataset_{generation.get()}"
                # Mount effect is responsible for module registration.
                # This render function only creates the UI markup.
                if mounted_generation.get() != generation.get():
                    return ui.div("Loading plotting workspace...")
                return ui_factories[name](name)

        return App(app_ui, server)

    def run(self, *, host="127.0.0.1", port=0, launch_browser=True, daemon=True):
        """Run PlotApp in a background thread, returning the thread."""
        from shiny import run_app
        app = self.create_app()

        def serve():
            run_app(app, host=host, port=port, launch_browser=launch_browser, dev_mode=False)

        thread = threading.Thread(target=serve, name="apmtools-plotapp", daemon=daemon)
        thread.start()
        return thread

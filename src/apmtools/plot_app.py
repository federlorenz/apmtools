"""Standalone, replaceable-dataset Shiny plotting application.

Pickle loading is intentionally restricted to explicitly selected local files.
Only open trusted pickle files: Python pickle can execute arbitrary code.
"""
from __future__ import annotations

from pathlib import Path
import pickle
import threading
import os
import tempfile


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


def save_summary_pickle(summary, destination, *, overwrite=False):
    """Atomically persist the current source Summary and retained plot presets."""
    from .classes import Summary
    if not isinstance(summary, Summary):
        raise TypeError("No Summary dataset is loaded")
    target = Path(destination).expanduser().resolve()
    if target.suffix.lower() != ".pkl":
        raise ValueError("The filename must end in .pkl")
    if not target.parent.is_dir():
        raise FileNotFoundError(f"Directory does not exist: {target.parent}")
    if target.exists() and not overwrite:
        raise FileExistsError(f"File already exists: {target}")
    if target.exists() and not target.is_file():
        raise ValueError(f"Destination is not a regular file: {target}")
    from .plot_configs import validate_saved_plots
    retained = validate_saved_plots(getattr(summary, "plots", None))
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode="wb", dir=target.parent,
                                         prefix=".apmtools-", suffix=".tmp",
                                         delete=False) as stream:
            temporary = Path(stream.name)
            snapshot = summary.copy(deep=False)
            snapshot.plots = retained
            pickle.dump(snapshot, stream, protocol=pickle.HIGHEST_PROTOCOL)
            stream.flush()
            os.fsync(stream.fileno())
        with temporary.open("rb") as stream:
            restored = pickle.load(stream)
        if getattr(restored, "plots", None) != retained:
            raise RuntimeError("Saved plot configurations failed pickle verification")
        if overwrite:
            os.replace(temporary, target)
        else:
            os.link(temporary, target)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    return target


class PlotApp:
    """Launch a Shiny plot workspace with an optional initial Summary.

    Examples
    --------
    >>> PlotApp(summary).run()
    >>> PlotApp().run()
    """

    def __init__(self, summary=None, *, x=None, y=None, group_by=None,
                 plot_type="line", title="Summary plot", backend="matplotlib"):
        from .classes import Summary
        if summary is not None and not isinstance(summary, Summary):
            raise TypeError("PlotApp expects an apmtools Summary or None")
        self.original_summary = None
        self.summary = None
        self.created_summary = None
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
        from .operation_log import OperationLog, stream_operation
        from .dataset_creation import scan_folder_async, scan_results_frame, scan_parameters, scan_directory_choices, parse_scan_levels, create_summary_async, summary_destination
        from .shiny.ui import _table_preview_frame
        import shinyswatch

        app_ui = ui.page_fluid(
            ui.tags.button("Hide toolbars", type="button", class_="btn btn-sm btn-outline-secondary my-0", id="standalone-toolbar-collapse-toggle", **{"aria-expanded": "true"}),
            ui.div(
                ui.input_action_button("open_dataset", "Open Dataset", class_="btn-outline-primary"),
                ui.output_ui("dataset_toolbar_actions"),
                ui.output_text("dataset_status", inline=True),
                ui.div(shinyswatch.theme_picker_ui(), class_="summary-app-theme ms-auto"),
                class_="d-flex align-items-center gap-1 my-0 standalone-dataset-toolbar",
            ),
            ui.tags.style("""
              body:has(#standalone-toolbar-collapse-toggle) #summary-dataset-toolbar {display:none !important;}
              .summary-empty-dataset-panels {display:grid; grid-template-columns:minmax(10rem, 1fr) minmax(0, 3fr) minmax(10rem, 1fr);
                gap:.75rem; height:max(420px, calc(100dvh - 170px)); min-height:420px;
                margin-top:.5rem; align-items:stretch;}
              .summary-empty-dataset-panels > div {border:1px solid var(--bs-border-color); border-radius:.5rem;
                min-height:0; min-width:0; background:var(--bs-body-bg); overflow:auto;}
              .summary-create-left {padding:.75rem; min-width:0;}
              .summary-create-main {padding:.75rem; min-width:0; display:flex; flex-direction:column;
                overflow:hidden !important;}
              .summary-create-main > .shiny-html-output {flex:1 1 auto; min-height:0; min-width:0;
                display:flex; flex-direction:column;}
              .summary-scan-results-content {flex:1 1 auto; min-height:0; min-width:0;
                display:flex; flex-direction:column; gap:.5rem;}
              .summary-scan-preview-toolbar {flex:0 0 auto;}
              .summary-scan-grid {flex:1 1 auto; height:100%; min-height:0; min-width:0;
                overflow-x:auto; overflow-y:auto;}
              .summary-scan-grid > * {height:100% !important; min-width:0;}
              .summary-scan-grid .shiny-data-grid {height:100% !important; min-width:100%; overflow:auto;}
              .summary-scan-grid .shiny-data-grid table {width:max-content; min-width:100%;}
              .summary-scan-grid .shiny-data-grid th,
              .summary-scan-grid .shiny-data-grid td {min-width:150px; white-space:nowrap;}
              .summary-scan-grid .shiny-data-grid input {min-width:115px;}
              .summary-folder-browser-list {max-height:min(50vh, 420px); min-height:180px;
                overflow-y:auto; overflow-x:hidden; border:1px solid var(--bs-border-color);
                border-radius:.5rem; padding:.4rem .75rem;}
              .summary-folder-browser-list .shiny-input-radiogroup {margin-bottom:0;}
              .summary-folder-browser-list .form-check {padding:.35rem .25rem .35rem 1.6rem;
                border-radius:.25rem;}
              .summary-folder-browser-list .form-check:hover {background:var(--bs-tertiary-bg);}
              .summary-folder-browser-list label {cursor:pointer; overflow-wrap:anywhere;}
              .summary-folder-choice {max-width:100%; overflow:hidden; text-overflow:ellipsis; white-space:nowrap;}
              .summary-scan-tab-label {color:#0c7690; font-weight:700;}
              .summary-create-left .nav-link:has(.summary-scan-tab-label) {
                background:#d9eef2; border-color:#acd5dc; border-radius:.4rem .4rem 0 0;
              }
              .summary-create-left .nav-link.active:has(.summary-scan-tab-label) {background:#bfe5ec;}
              .summary-folder-browser-current {overflow-wrap:anywhere; font-size:.875rem;}
              .summary-folder-browser-actions {display:flex; gap:.5rem; align-items:center; flex-wrap:wrap;}
              .summary-operation-log {flex:0 0 auto; margin-top:.5rem;}
              .summary-log-resize-handle {width:100%; min-height:18px; margin-bottom:.25rem;
                display:flex; align-items:center; justify-content:center; cursor:ns-resize;
                touch-action:none; user-select:none; border-radius:.3rem;
                background:var(--bs-tertiary-bg); border:1px solid var(--bs-border-color);}
              .summary-log-resize-handle::after {content:""; width:64px; height:4px;
                border-top:2px solid var(--bs-secondary-color);
                border-bottom:2px solid var(--bs-secondary-color);}
              .summary-log-resize-handle:hover, .summary-log-resize-handle:focus-visible {
                background:var(--bs-secondary-bg); outline:2px solid var(--bs-primary); outline-offset:1px;}
              .summary-operation-log pre {height:210px; min-height:70px; max-height:80dvh;
                resize:none; overflow:auto; box-sizing:border-box;
                margin:0; white-space:pre-wrap; overflow-wrap:anywhere; font-size:.78rem;
                background:var(--bs-tertiary-bg); border-radius:.35rem; padding:.65rem;}
              .summary-create-right {min-width:0;}
              @media(max-width:850px){.summary-empty-dataset-panels {grid-template-columns:1fr;
                height:auto; min-height:0;}
                .summary-empty-dataset-panels > div {min-height:15vh;}
                .summary-create-main {height:max(400px, calc(100dvh - 180px));}}
            """),
            ui.tags.script("""
              // Full-width resize grip; no Shiny reactive values or polling needed.
              // Grabbing the top grip and moving upward makes the log taller.
              (function () {
                let active = null;
                function heightLimit() { return Math.max(70, Math.floor(window.innerHeight * 0.8)); }
                function setHeight(log, height) {
                  log.style.height = Math.max(70, Math.min(heightLimit(), Math.round(height))) + 'px';
                }
                document.addEventListener('pointerdown', function(event) {
                  const handle = event.target.closest && event.target.closest('#summary-log-resize-handle');
                  if (!handle || event.button !== 0) return;
                  const log = document.getElementById('summary-operation-log-scroll');
                  if (!log) return;
                  active = {pointerId:event.pointerId, handle:handle, log:log,
                    startY:event.clientY, startHeight:log.getBoundingClientRect().height};
                  handle.setPointerCapture(event.pointerId);
                  event.preventDefault();
                });
                document.addEventListener('pointermove', function(event) {
                  if (!active || event.pointerId !== active.pointerId) return;
                  if (!active.log.isConnected) { active = null; return; }
                  setHeight(active.log, active.startHeight + active.startY - event.clientY);
                  event.preventDefault();
                });
                function stopResize(event) {
                  if (active && event.pointerId === active.pointerId) active = null;
                }
                document.addEventListener('pointerup', stopResize);
                document.addEventListener('pointercancel', stopResize);
                document.addEventListener('keydown', function(event) {
                  const grip = event.target.closest && event.target.closest('#summary-log-resize-handle');
                  if (!grip || !['ArrowUp', 'ArrowDown'].includes(event.key)) return;
                  const log = document.getElementById('summary-operation-log-scroll');
                  if (!log) return;
                  const increment = event.shiftKey ? 50 : 20;
                  setHeight(log, log.getBoundingClientRect().height + (event.key === 'ArrowUp' ? increment : -increment));
                  event.preventDefault();
                });
              })();
              document.addEventListener('shiny:value', function(event) {
                if (event.target && event.target.id === 'operation_log_text') {
                  requestAnimationFrame(function() {
                    const box = document.getElementById('summary-operation-log-scroll');
                    if (box) box.scrollTop = box.scrollHeight;
                  });
                }
              });
            """),
            ui.output_ui("dataset_workspace"),
            title="PlotApp",
        )

        def server(input, output, session):
            dataset = reactive.value(self.summary)
            generation = reactive.value(0)
            blank_tabs = reactive.value(())
            selected_blank_tab = reactive.value(None)
            status = reactive.value("" if self.summary is not None else "No dataset loaded")
            scan_folder_path = reactive.value("")
            scan_browse_path = reactive.value(Path.home().resolve())
            scan_metadata = reactive.value({})
            scan_busy = reactive.value(False)
            operation_messages = reactive.value(())
            scan_output = reactive.value(None)
            generated_summary = reactive.value(None)
            scan_completed = reactive.value(False)
            summary_filename = reactive.value("")
            summary_busy = reactive.value(False)
            create_epoch = reactive.value(0)
            scanned_dataset = None
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

            @render.ui
            def dataset_toolbar_actions():
                if dataset.get() is not None:
                    return ui.div(
                        ui.input_action_button("save_dataset_root", "Save dataset", class_="btn-outline-primary"),
                        ui.input_action_button("close_dataset", "Close dataset", class_="btn-outline-danger"),
                        class_="d-flex gap-1 align-items-center",
                    )
                return ui.div(
                    ui.input_action_button("create_dataset", "Create dataset", class_="btn-outline-primary"),
                    ui.input_action_button("modify_dataset", "Modify dataset", class_="btn-outline-primary"),
                    class_="d-flex gap-1 align-items-center",
                )

            @reactive.effect
            @reactive.event(input.create_dataset)
            def create_empty_workspace():
                if dataset.get() is None:
                    tabs = blank_tabs.get()
                    if "create" not in tabs:
                        blank_tabs.set((*tabs, "create"))
                    selected_blank_tab.set("create")

            @reactive.effect
            @reactive.event(input.modify_dataset)
            def modify_empty_workspace():
                if dataset.get() is None:
                    tabs = blank_tabs.get()
                    if "modify" not in tabs:
                        blank_tabs.set((*tabs, "modify"))
                    selected_blank_tab.set("modify")

            @reactive.effect
            @reactive.event(input.close_create_dataset_tab)
            def close_create_dataset_tab():
                nonlocal scanned_dataset
                if "create" not in blank_tabs.get():
                    return
                # Invalidate pending async scan/summary completions; releasing
                # the session references discards the underlying Dataset.
                create_epoch.set(create_epoch.get() + 1)
                scanned_dataset = None
                self.created_summary = None
                scan_output.set(None)
                generated_summary.set(None)
                scan_completed.set(False)
                scan_metadata.set({})
                scan_folder_path.set("")
                summary_filename.set("")
                scan_busy.set(False)
                summary_busy.set(False)
                operation_messages.set(())
                remaining = tuple(tab for tab in blank_tabs.get() if tab != "create")
                blank_tabs.set(remaining)
                selected_blank_tab.set(remaining[-1] if remaining else None)
                status.set("No dataset loaded")

            def append_operation_messages(new_messages, epoch):
                # An operation may finish after its Create dataset tab was closed.
                if epoch != create_epoch.get() or "create" not in blank_tabs.get():
                    return
                operation_messages.set((operation_messages.get() + tuple(new_messages))[-2000:])

            @render.text
            def operation_log_text():
                return "\n".join(operation_messages.get())

            @reactive.effect
            @reactive.event(input.clear_operation_log)
            def clear_operation_log():
                operation_messages.set(())

            @render.ui
            def create_summary_tab_content():
                if not scan_completed.get():
                    return ui.div()
                parts = [
                    ui.input_switch("create_summary_all_variables", "all_variables", value=False),
                    ui.output_ui("create_summary_run_control"),
                ]
                if generated_summary.get() is not None:
                    parts.extend([
                        ui.input_action_button(
                            "create_summary_choose_filename", "Save to file",
                            class_="btn-outline-primary w-100",
                        ),
                        ui.input_checkbox("create_summary_overwrite", "Overwrite file", value=False),
                    ])
                return ui.div(*parts)

            @reactive.effect
            @reactive.event(input.create_summary_choose_filename)
            def create_summary_filename_dialog():
                if generated_summary.get() is None:
                    return
                ui.modal_show(ui.modal(
                    ui.input_text("create_summary_filename_text", "Summary .pkl destination on the Shiny computer",
                                  value=summary_filename.get() or "summary.pkl"),
                    title="Save Summary to file", easy_close=True,
                    footer=ui.div(ui.modal_button("Cancel"),
                                  ui.input_action_button("create_summary_set_filename", "Save", class_="btn-primary")),
                ))

            @reactive.effect
            @reactive.event(input.create_summary_set_filename)
            def set_create_summary_filename():
                summary = generated_summary.get()
                if summary is None or "create" not in blank_tabs.get():
                    return
                try:
                    overwrite = bool(input.create_summary_overwrite() or False)
                    target = summary_destination(input.create_summary_filename_text(), overwrite=overwrite)
                    saved = save_summary_pickle(summary, target, overwrite=overwrite)
                except Exception as exc:
                    ui.notification_show(f"Could not save Summary: {exc}", type="error", duration=10)
                    return
                summary_filename.set(str(saved))
                status.set(f"Summary saved: {saved}")
                ui.modal_remove()
                ui.notification_show(f"Summary saved to {saved}", type="message", duration=8)

            @render.ui
            def create_summary_run_control():
                ready = (scan_completed.get() and scanned_dataset is not None
                         and generated_summary.get() is None
                         and not scan_busy.get() and not summary_busy.get())
                return ui.input_action_button(
                    "create_summary_run", "Run", disabled=not ready,
                    class_="btn-primary w-100" if ready else "btn-outline-secondary w-100",
                )

            @reactive.effect
            @reactive.event(input.create_summary_run)
            async def run_create_summary():
                if (not scan_completed.get() or scanned_dataset is None or scan_busy.get()
                        or summary_busy.get() or generated_summary.get() is not None
                        or "create" not in blank_tabs.get()):
                    return
                epoch = create_epoch.get()
                source_dataset = scanned_dataset
                all_variables = bool(input.create_summary_all_variables() or False)
                summary_busy.set(True)
                status.set("Creating Summary…")
                try:
                    log = OperationLog()
                    log.emit("Creating Summary…")
                    summary = await stream_operation(
                        create_summary_async(source_dataset, all_variables=all_variables,
                                             message_callback=log.emit),
                        log, lambda messages: append_operation_messages(messages, epoch),
                    )
                    if epoch != create_epoch.get() or "create" not in blank_tabs.get():
                        return
                    self.created_summary = summary
                    generated_summary.set(summary)
                    append_operation_messages(["Summary created successfully."], epoch)
                    status.set(f"Summary created: {len(summary):,} rows")
                    ui.notification_show("Summary created. Use Save to file to save a .pkl.",
                                         type="message", duration=8)
                except Exception as exc:
                    if epoch == create_epoch.get():
                        append_operation_messages([f"Summary creation failed: {exc}"], epoch)
                        status.set("Summary creation failed")
                        ui.notification_show(f"Summary creation failed: {exc}", type="error", duration=12)
                finally:
                    if epoch == create_epoch.get():
                        summary_busy.set(False)

            @render.ui
            def scan_folder_button():
                location = scan_folder_path.get()
                return ui.input_action_button(
                    "scan_choose_folder", location or "Folder",
                    class_="btn-outline-primary summary-folder-choice w-100",
                    title=location or "Select scan folder",
                )

            @reactive.effect
            @reactive.event(input.scan_choose_folder)
            def scan_folder_dialog():
                # Server-side directory navigation: a browser cannot provide an
                # absolute server directory through a native OS folder picker.
                existing = scan_folder_path.get()
                initial = Path(existing).expanduser() if existing else Path.home()
                scan_browse_path.set(initial.resolve() if initial.is_dir() else Path.home().resolve())
                ui.modal_show(ui.modal(
                    ui.p("Browse folders on the computer running Shiny.", class_="small text-muted"),
                    ui.output_ui("scan_folder_browser"),
                    title="Select scan folder", easy_close=True, size="l",
                    footer=ui.div(
                        ui.modal_button("Cancel"),
                        ui.input_action_button("scan_set_folder", "Select this folder", class_="btn-primary")),
                ))

            @render.ui
            def scan_folder_browser():
                current = scan_browse_path.get()
                try:
                    directories = scan_directory_choices(current)
                except (OSError, PermissionError) as exc:
                    return ui.div(
                        ui.p(f"Cannot read this directory: {exc}", class_="text-danger"),
                        ui.input_action_button("scan_browse_up", "Up one folder", class_="btn-outline-secondary"),
                    )
                return ui.div(
                    ui.p("Current folder", class_="small text-muted mb-1"),
                    ui.div(str(current), class_="summary-folder-browser-current fw-semibold mb-3"),
                    ui.div(
                        ui.input_radio_buttons(
                            "scan_browse_child", None,
                            choices={name: f"📁 {name}" for name in directories},
                            selected=directories[0] if directories else None,
                        ),
                        ui.p("No subfolders", class_="text-muted m-2") if not directories else ui.div(),
                        class_="summary-folder-browser-list mb-3",
                    ),
                    ui.div(
                        ui.input_action_button("scan_browse_home", "Home", class_="btn-outline-secondary"),
                        ui.input_action_button("scan_browse_up", "Up one folder", class_="btn-outline-secondary",
                                               disabled=current == current.parent),
                        ui.input_action_button("scan_browse_open", "Open selected folder", class_="btn-outline-primary",
                                               disabled=not directories),
                        class_="summary-folder-browser-actions",
                    ),
                    ui.p("Browse folders on the computer running Shiny. Select this folder to scan it.",
                         class_="small text-muted mt-3"),
                )

            @reactive.effect
            @reactive.event(input.scan_browse_home)
            def scan_browse_home():
                scan_browse_path.set(Path.home().resolve())

            @reactive.effect
            @reactive.event(input.scan_browse_up)
            def scan_browse_up():
                current = scan_browse_path.get()
                scan_browse_path.set(current.parent)

            @reactive.effect
            @reactive.event(input.scan_browse_open)
            def scan_browse_open():
                child = str(input.scan_browse_child() or "")
                if not child or child in (".", "..") or Path(child).name != child:
                    return
                target = scan_browse_path.get() / child
                if not target.is_dir():
                    ui.notification_show("The selected folder is no longer available", type="warning")
                    return
                scan_browse_path.set(target.resolve())

            @reactive.effect
            @reactive.event(input.scan_set_folder)
            def set_scan_folder():
                folder = scan_browse_path.get()
                if not folder.is_dir():
                    ui.notification_show("Choose an existing folder", type="warning")
                    return
                scan_folder_path.set(str(folder))
                ui.modal_remove()

            @render.ui
            def scan_levels_ui():
                return ui.input_text(
                    "scan_levels", "Levels (comma separated)",
                    placeholder="study, monitor, location",
                )

            @reactive.effect
            @reactive.event(input.scan_add_metadata_dialog)
            def add_scan_metadata_dialog():
                ui.modal_show(ui.modal(
                    ui.input_text("scan_metadata_category", "Metadata category"),
                    ui.input_text("scan_metadata_value", "Metadata value"),
                    title="Add metadata", easy_close=True,
                    footer=ui.div(ui.modal_button("Close"),
                                  ui.input_action_button("scan_metadata_add", "Add", class_="btn-primary")),
                ))

            @reactive.effect
            @reactive.event(input.scan_metadata_add)
            def add_scan_metadata():
                key = str(input.scan_metadata_category() or "").strip()
                if not key:
                    ui.notification_show("Enter a metadata category", type="warning")
                    return
                value = str(input.scan_metadata_value() or "")
                current = dict(scan_metadata.get())
                current[key] = value
                scan_metadata.set(current)
                ui.update_text("scan_metadata_category", value="")
                ui.update_text("scan_metadata_value", value="")

            @render.ui
            def scan_metadata_list():
                metadata = scan_metadata.get()
                if not metadata:
                    return ui.div()
                return ui.div(ui.strong("Metadata"),
                              ui.tags.ul(*(ui.tags.li(f"{key}: {val}")
                                           for key, val in metadata.items())),
                              class_="small mt-2")

            def current_scan_levels():
                return parse_scan_levels(input.scan_levels() or "")

            @render.ui
            def scan_start_control():
                levels = current_scan_levels()
                ready = (bool(scan_folder_path.get()) and bool(levels)
                         and all(levels) and not scan_busy.get())
                return ui.input_action_button("scan_start", "Start scanning",
                                              class_="btn-primary w-100" if ready else "btn-outline-secondary w-100",
                                              disabled=not ready)

            @reactive.effect
            @reactive.event(input.scan_start)
            async def start_scan():
                nonlocal scanned_dataset
                if (dataset.get() is not None or scan_busy.get() or summary_busy.get()
                        or "create" not in blank_tabs.get()):
                    return
                epoch = create_epoch.get()
                levels = current_scan_levels()
                try:
                    # Fail quickly before scheduling the potentially slow scan.
                    params = scan_parameters(scan_folder_path.get(), levels,
                                             input.scan_timezone(), scan_metadata.get())
                except Exception as exc:
                    ui.notification_show(f"Cannot start scan: {exc}", type="error", duration=10)
                    return
                scan_busy.set(True)
                status.set("Scanning folder…")
                try:
                    # Stream only the current operation's messages, without touching
                    # process-global stdout or other Shiny sessions.
                    log = OperationLog()
                    log.emit(f"Scanning folder: {params['directory']}")
                    scan_options = {"message_callback": log.emit}
                    if scanned_dataset is not None:
                        scan_options["dataset"] = scanned_dataset
                    result = await stream_operation(
                        scan_folder_async(params["directory"], params["levels"],
                                          params["utc_timezone"], params["add_m"],
                                          **scan_options),
                        log, lambda messages: append_operation_messages(messages, epoch),
                    )
                    if epoch != create_epoch.get() or "create" not in blank_tabs.get():
                        return
                    frame = scan_results_frame(result)
                    scanned_dataset = result
                    scan_completed.set(True)
                    generated_summary.set(None)
                    self.created_summary = None
                    # Metadata applies to the just-completed scan only;
                    # the scanned Dataset retains metadata already attached to entries.
                    scan_metadata.set({})
                    scan_output.set(frame)
                    status.set(f"Scan complete: {len(frame)} monitor entries")
                    append_operation_messages([f"Scan complete: {len(frame)} monitor entries"], epoch)
                    if frame.empty:
                        ui.notification_show("Scan complete: no recognized monitor files found",
                                             type="warning", duration=10)
                except Exception as exc:
                    if epoch == create_epoch.get():
                        append_operation_messages([f"Scan failed: {exc}"], epoch)
                        status.set("Scan failed")
                        ui.notification_show(f"Scan failed: {exc}", type="error", duration=12)
                finally:
                    if epoch == create_epoch.get():
                        scan_busy.set(False)

            @render.ui
            def scan_result_content():
                if scan_busy.get():
                    return ui.div(ui.p("Scanning folder…"), class_="p-3")
                if summary_busy.get():
                    return ui.div(ui.p("Creating Summary…"), class_="p-3")
                summary = generated_summary.get()
                frame = summary if summary is not None else scan_output.get()
                if frame is None:
                    return ui.div()
                return ui.div(
                    ui.div(ui.strong(f"Summary rows: {len(frame):,}" if summary is not None
                                     else f"Scanned monitors: {len(frame):,}"),
                           ui.input_select("scan_preview_rows", "Rows in grid preview",
                                           choices={"500": "500 rows", "1000": "1,000 rows",
                                                    "5000": "5,000 rows", "all": "All rows"},
                                           selected="1000"),
                           class_="d-flex align-items-center justify-content-between gap-3 flex-wrap summary-scan-preview-toolbar"),
                    ui.div(ui.output_data_frame("scan_results_grid"), class_="summary-scan-grid"),
                    class_="summary-scan-results-content",
                )

            @render.data_frame
            def scan_results_grid():
                from shiny import render as shiny_render
                summary = generated_summary.get()
                frame = summary if summary is not None else scan_output.get()
                if frame is None:
                    return shiny_render.DataGrid(__import__("pandas").DataFrame(), width="100%")
                try:
                    row_count = input.scan_preview_rows() or "1000"
                except Exception:
                    row_count = "1000"
                preview = _table_preview_frame(frame, row_count).copy()
                # Mixed metadata objects must be browser-safe strings.
                for name in preview.columns:
                    if preview[name].dtype == object:
                        preview[name] = preview[name].map(
                            lambda value: value if isinstance(value, (str, int, float, bool)) or value is None
                            else str(value))
                return shiny_render.DataGrid(preview, filters=True, width="100%", height="100%")

            @reactive.effect
            @reactive.event(input.close_dataset)
            def confirm_close_dataset():
                if dataset.get() is None:
                    return
                ui.modal_show(ui.modal(
                    ui.p("Close the current dataset? Unsaved dataset changes and plot settings may be lost."),
                    title="Close dataset",
                    footer=ui.div(ui.modal_button("Cancel"),
                                  ui.input_action_button("confirm_close_dataset", "Close dataset", class_="btn-danger")),
                ))

            @reactive.effect
            @reactive.event(input.confirm_close_dataset)
            def close_current_dataset():
                if dataset.get() is None:
                    return
                self.original_summary = None
                self.summary = None
                dataset.set(None)
                generation.set(generation.get() + 1)
                blank_tabs.set(())
                selected_blank_tab.set(None)
                status.set("No dataset loaded")
                ui.modal_remove()

            @reactive.effect
            @reactive.event(input.save_dataset_root)
            def show_root_save_dialog():
                if dataset.get() is None or self.original_summary is None:
                    ui.notification_show("Open a dataset before saving", type="warning")
                    return
                ui.modal_show(ui.modal(
                    ui.p("Save the loaded Summary and its saved plot configurations as a pickle."),
                    ui.input_text("root_dataset_path", "Destination path on the computer running Shiny",
                                  value="summary.pkl"),
                    ui.input_checkbox("root_dataset_overwrite", "Overwrite existing file", value=False),
                    title="Save dataset",
                    footer=ui.div(ui.modal_button("Cancel"),
                                  ui.input_action_button("root_confirm_save", "Save pickle", class_="btn-primary")),
                ))

            @reactive.effect
            @reactive.event(input.root_confirm_save)
            def save_root_dataset():
                try:
                    target = save_summary_pickle(
                        self.original_summary, input.root_dataset_path(),
                        overwrite=bool(input.root_dataset_overwrite()))
                    ui.modal_remove()
                    ui.notification_show(f"Dataset saved to {target}", type="message", duration=8)
                except Exception as exc:
                    ui.notification_show(f"Could not save dataset: {exc}", type="error", duration=10)

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
                nonlocal scanned_dataset
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
                create_epoch.set(create_epoch.get() + 1)
                blank_tabs.set(())
                selected_blank_tab.set(None)
                scanned_dataset = None
                self.created_summary = None
                generated_summary.set(None)
                scan_completed.set(False)
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
                    tabs = blank_tabs.get()
                    if not tabs:
                        return ui.div(ui.h3("No dataset loaded"),
                                      ui.p("Open a Summary .pkl file, or select Create dataset or Modify dataset."),
                                      class_="p-4")
                    labels = {"create": "Create dataset", "modify": "Modify dataset"}
                    return ui.navset_tab(
                        *(ui.nav_panel(
                            (ui.tags.span(
                                labels[mode],
                                ui.input_action_button("close_create_dataset_tab", "×",
                                                       class_="btn btn-sm btn-link text-danger p-0 ms-2",
                                                       title="Close Create dataset tab")
                                if mode == "create" else ui.div(),
                                class_="d-inline-flex align-items-center") if mode == "create" else labels[mode]),
                            ui.div(
                                (ui.div(
                                    ui.navset_tab(
                                        *( [ui.nav_panel(ui.tags.span("Scan folder", class_="summary-scan-tab-label"),
                                            ui.output_ui("scan_folder_button"),
                                            ui.output_ui("scan_levels_ui"),
                                            ui.input_numeric("scan_timezone", "UTC timezone", value=0, step=0.5),
                                            ui.input_action_button("scan_add_metadata_dialog", "Add metadata",
                                                                   class_="btn-outline-secondary w-100"),
                                            ui.output_ui("scan_metadata_list"),
                                            ui.output_ui("scan_start_control"),
                                        )] if generated_summary.get() is None else [] ),
                                        *( [ui.nav_panel("Create summary", ui.output_ui("create_summary_tab_content"))]
                                           if scan_completed.get() else [] ),
                                    ), class_="summary-empty-dataset-left summary-create-left")
                                 if mode == "create" else ui.div(class_="summary-empty-dataset-left")),
                                (ui.div(
                                    ui.output_ui("scan_result_content"),
                                    ui.div(ui.accordion(
                                        ui.accordion_panel("Operation log",
                                            ui.div(
                                                ui.input_action_button("clear_operation_log", "Clear log",
                                                                       class_="btn-outline-secondary btn-sm mb-2"),
                                                ui.tags.div(id="summary-log-resize-handle",
                                                            class_="summary-log-resize-handle",
                                                            role="separator", tabindex="0",
                                                            **{"aria-label": "Drag to resize operation log height; use Up and Down arrow keys"},
                                                            title="Drag this bar up or down to resize the log"),
                                                ui.tags.pre(ui.output_text("operation_log_text", inline=True),
                                                            id="summary-operation-log-scroll"),
                                                ui.tags.small("Drag the horizontal handle above the log, or focus it and press Up/Down.",
                                                              class_="text-body-secondary"),
                                            ),
                                        ), open=False), class_="summary-operation-log"),
                                    class_="summary-empty-dataset-main summary-create-main")
                                 if mode == "create" else ui.div(class_="summary-empty-dataset-main")),
                                ui.div(class_="summary-empty-dataset-right summary-create-right" if mode == "create" else "summary-empty-dataset-right"),
                                class_="summary-empty-dataset-panels",
                            ),
                            value=mode,
                        ) for mode in tabs),
                        id="blank_dataset_tabs",
                        selected=selected_blank_tab.get(),
                    )
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

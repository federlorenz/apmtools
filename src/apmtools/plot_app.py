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
        from .dataset_creation import scan_folder_async, scan_results_frame, scan_parameters, scan_directory_choices, parse_scan_levels, multiple_custom_monitor_signatures, create_summary_async, summary_destination
        from .shiny.ui import _table_preview_frame
        from . import dataset_editing as editing
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
              /* Edge-toggle sidebars with draggable widths; resize affects only this workspace. */
              .summary-modify-panels {display:grid;
                grid-template-columns:var(--modify-left-width, 300px) minmax(0, 1fr) var(--modify-right-width, 320px);
                gap:0; margin-top:.5rem; height:max(420px, calc(100dvh - 210px));
                min-height:420px; align-items:stretch;}
              .summary-modify-panels > div {min-width:0; min-height:0; background:var(--bs-body-bg);
                border:1px solid var(--bs-border-color);}
              .summary-modify-left, .summary-modify-right {position:relative; overflow:visible !important;}
              .summary-modify-sidebar-body {height:100%; overflow:auto; padding:.75rem;}
              .summary-modify-main {overflow:hidden !important;}
              .summary-modify-edge-toggle {position:absolute; z-index:6; top:12px;
                width:23px; height:35px; padding:0; line-height:1; display:flex;
                align-items:center; justify-content:center; background:var(--bs-body-bg);
                color:var(--bs-body-color); border:1px solid var(--bs-border-color);
                border-radius:.35rem; box-shadow:0 1px 4px rgba(0,0,0,.12);}
              .summary-modify-left .summary-modify-edge-toggle {right:-12px;}
              .summary-modify-right .summary-modify-edge-toggle {left:-12px;}
              .summary-modify-edge-toggle:hover, .summary-modify-edge-toggle:focus-visible {
                background:var(--bs-secondary-bg); outline:2px solid var(--bs-primary); outline-offset:1px;}
              .summary-modify-resize-grip {position:absolute; z-index:5; top:54px; bottom:0;
                width:10px; cursor:col-resize; touch-action:none; user-select:none;}
              .summary-modify-left .summary-modify-resize-grip {right:-5px;}
              .summary-modify-right .summary-modify-resize-grip {left:-5px;}
              .summary-modify-resize-grip::after {content:""; position:absolute; top:0; bottom:0;
                left:4px; border-left:2px solid var(--bs-border-color);}
              .summary-modify-resize-grip:hover::after {border-color:var(--bs-primary);}
              .summary-modify-left-hidden {--modify-left-width:35px !important;}
              .summary-modify-right-hidden {--modify-right-width:35px !important;}
              .summary-modify-left-hidden .summary-modify-left .summary-modify-sidebar-body,
              .summary-modify-right-hidden .summary-modify-right .summary-modify-sidebar-body,
              .summary-modify-left-hidden .summary-modify-left .summary-modify-resize-grip,
              .summary-modify-right-hidden .summary-modify-right .summary-modify-resize-grip {display:none;}
              @media(max-width:850px) {
                .summary-modify-panels {grid-template-columns:minmax(0,1fr) !important;
                  height:auto; min-height:420px; gap:.5rem;}
                .summary-modify-left-hidden .summary-modify-left,
                .summary-modify-right-hidden .summary-modify-right {min-height:35px;}
                .summary-modify-resize-grip {display:none;}
                .summary-modify-main {min-height:400px;}
              }
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
              .summary-create-panels {grid-template-columns:minmax(10rem, 1fr) minmax(0, 3fr);}
              @media(max-width:850px){.summary-empty-dataset-panels {grid-template-columns:1fr;
                height:auto; min-height:0;}
                .summary-empty-dataset-panels > div {min-height:15vh;}
                .summary-create-main {height:max(400px, calc(100dvh - 180px));}}
            """),
            ui.tags.script("""
              // Delegated handlers work even if Shiny replaces the Modify tab.
              const modifySidebarSizes = {left:300, right:320};
              const modifySidebarClosed = {left:false, right:false};
              function applyModifySidebarState(panels) {
                if (!panels) return;
                for (const side of ['left', 'right']) {
                  panels.style.setProperty('--modify-' + side + '-width', modifySidebarSizes[side] + 'px');
                  panels.classList.toggle('summary-modify-' + side + '-hidden', modifySidebarClosed[side]);
                  const toggle = panels.querySelector('.summary-modify-' + side + ' .summary-modify-edge-toggle');
                  if (!toggle) continue;
                  const collapsed = modifySidebarClosed[side];
                  toggle.textContent = side === 'left' ? (collapsed ? '›' : '‹') : (collapsed ? '‹' : '›');
                  toggle.setAttribute('aria-expanded', String(!collapsed));
                  toggle.setAttribute('aria-label', (collapsed ? 'Show ' : 'Collapse ') + side + ' sidebar');
                  toggle.title = toggle.getAttribute('aria-label');
                }
              }
              document.addEventListener('click', function(event) {
                const button = event.target.closest && event.target.closest('.summary-modify-edge-toggle');
                if (!button) return;
                const side = button.getAttribute('data-side');
                if (side !== 'left' && side !== 'right') return;
                modifySidebarClosed[side] = !modifySidebarClosed[side];
                applyModifySidebarState(button.closest('.summary-modify-panels'));
              });
              // Keep sizes across Modify tab remounts, without rerunning Shiny inputs.
              const observeModifyPanels = new MutationObserver(function() {
                for (const panel of document.querySelectorAll('.summary-modify-panels')) {
                  if (panel.dataset.modifyInitialized === 'true') continue;
                  panel.dataset.modifyInitialized = 'true';
                  applyModifySidebarState(panel);
                }
              });
              observeModifyPanels.observe(document.documentElement, {childList:true, subtree:true});
              let modifyDrag = null;
              document.addEventListener('pointerdown', function(event) {
                const grip = event.target.closest && event.target.closest('.summary-modify-resize-grip');
                if (!grip || event.button !== 0) return;
                const side = grip.getAttribute('data-side');
                const panels = grip.closest('.summary-modify-panels');
                if (!panels || modifySidebarClosed[side] || window.matchMedia('(max-width:850px)').matches) return;
                modifyDrag = {side, grip, panels, pointerId:event.pointerId,
                  startX:event.clientX, startWidth:modifySidebarSizes[side]};
                grip.setPointerCapture(event.pointerId);
                event.preventDefault();
              });
              document.addEventListener('pointermove', function(event) {
                if (!modifyDrag || event.pointerId !== modifyDrag.pointerId) return;
                const d = modifyDrag;
                if (!d.panels.isConnected) {modifyDrag=null; return;}
                const direction = d.side === 'left' ? 1 : -1;
                const other = d.side === 'left' ? 'right' : 'left';
                const otherWidth = modifySidebarClosed[other] ? 35 : modifySidebarSizes[other];
                const maxWidth = Math.max(160, d.panels.clientWidth - otherWidth - 240);
                modifySidebarSizes[d.side] = Math.min(maxWidth, Math.max(160,
                  Math.round(d.startWidth + direction * (event.clientX - d.startX))));
                applyModifySidebarState(d.panels);
              });
              function endModifyDrag(event) {
                if (!modifyDrag || event.pointerId !== modifyDrag.pointerId) return;
                modifyDrag=null;
              }
              document.addEventListener('pointerup', endModifyDrag);
              document.addEventListener('pointercancel', endModifyDrag);
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
            modify_open = reactive.value(False)
            plot_open = reactive.value(False)
            modify_timeline = reactive.value(None)
            selected_loaded_tab = reactive.value("plots")
            status = reactive.value("" if self.summary is not None else "No dataset loaded")
            scan_folder_path = reactive.value("")
            scan_browse_path = reactive.value(Path.home().resolve())
            scan_metadata = reactive.value({})
            scan_custom_count = reactive.value(1)
            modify_map_condition_count = reactive.value(1)
            modify_map_source_ref = [None]
            scan_busy = reactive.value(False)
            operation_messages = reactive.value(())
            scan_output = reactive.value(None)
            generated_summary = reactive.value(None)
            scan_completed = reactive.value(False)
            upas_filter_busy = reactive.value(False)
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
                # The Summary produced in Create dataset is available before
                # the user elects to open the Plot/Table workspace.
                has_summary = dataset.get() is not None or generated_summary.get() is not None
                if has_summary:
                    return ui.div(
                        ui.input_action_button("modify_dataset", "Modify Dataset", class_="btn-outline-primary"),
                        ui.input_action_button("save_dataset_root", "Save Dataset", class_="btn-outline-primary"),
                        ui.input_action_button("close_dataset", "Close Dataset", class_="btn-outline-danger"),
                        ui.input_action_button("plot_table_dataset", "Plot/Table Dataset", class_="btn-outline-primary"),
                        class_="d-flex gap-1 align-items-center",
                    )
                return ui.div(
                    ui.input_action_button("create_dataset", "Create dataset", class_="btn-outline-primary"),
                    class_="d-flex gap-1 align-items-center",
                )

            @reactive.effect
            @reactive.event(input.plot_table_dataset)
            def activate_plot_table_workspace():
                nonlocal scanned_dataset
                data = dataset.get()
                if data is None:
                    data = generated_summary.get()
                    if data is None:
                        return
                    # Promote an in-memory Create dataset result into the
                    # normal loaded Summary lifecycle only on explicit request.
                    self._set_dataset(data)
                    dataset.set(self.summary)
                    modify_timeline.set(None)
                    modify_open.set(False)
                    generation.set(generation.get() + 1)
                    create_epoch.set(create_epoch.get() + 1)
                    blank_tabs.set(())
                    selected_blank_tab.set(None)
                    scanned_dataset = None
                    scan_output.set(None)
                    scan_completed.set(False)
                    generated_summary.set(None)
                    self.created_summary = None
                plot_open.set(True)
                selected_loaded_tab.set("plots")

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
                if dataset.get() is not None:
                    modify_open.set(True)
                    selected_loaded_tab.set("modify")
                else:
                    tabs = blank_tabs.get()
                    if "modify" not in tabs:
                        blank_tabs.set((*tabs, "modify"))
                    selected_blank_tab.set("modify")

            @reactive.effect
            @reactive.event(input.loaded_dataset_tabs)
            def track_loaded_dataset_tab():
                """Respect direct navigation between the plot and Modify tabs."""
                selection = input.loaded_dataset_tabs()
                if selection in ("plots", "modify") and selected_loaded_tab.get() != selection:
                    selected_loaded_tab.set(selection)

            @reactive.effect
            @reactive.event(input.close_modify_dataset_tab)
            def close_modify_dataset_tab():
                if dataset.get() is not None:
                    modify_open.set(False)
                    selected_loaded_tab.set("plots")
                else:
                    remaining = tuple(tab for tab in blank_tabs.get() if tab != "modify")
                    blank_tabs.set(remaining)
                    selected_blank_tab.set(remaining[-1] if remaining else None)

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
                upas_filter_busy.set(False)
                scan_metadata.set({})
                scan_custom_count.set(1)
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
                         and not scan_busy.get() and not summary_busy.get()
                         and not upas_filter_busy.get())
                return ui.input_action_button(
                    "create_summary_run", "Run", disabled=not ready,
                    class_="btn-primary w-100" if ready else "btn-outline-secondary w-100",
                )

            @reactive.effect
            @reactive.event(input.create_summary_run)
            async def run_create_summary():
                if (not scan_completed.get() or scanned_dataset is None or scan_busy.get()
                        or summary_busy.get() or upas_filter_busy.get()
                        or generated_summary.get() is not None
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

            @render.ui
            def scan_custom_timestamp_format():
                if input.scan_custom_use_format():
                    return ui.input_text(
                        "scan_custom_date_format", "Custom timestamp format",
                        placeholder="%Y-%m-%d %H:%M:%S",
                    )
                return ui.div()

            def scan_custom_field(index, field):
                """The first row retains its original Shiny input IDs."""
                suffix = "" if index == 0 else f"_{index}"
                return input[f"scan_custom_{field}{suffix}"]()

            def current_custom_monitor_signatures():
                entries = []
                for index in range(scan_custom_count.get()):
                    use_format = bool(scan_custom_field(index, "use_format"))
                    entries.append({
                        "name": scan_custom_field(index, "name"),
                        "signature": scan_custom_field(index, "signature"),
                        "time_column": scan_custom_field(index, "time_column"),
                        "custom_format": use_format,
                        "date_format": (scan_custom_field(index, "date_format")
                                        if use_format else None),
                    })
                return multiple_custom_monitor_signatures(entries)

            def extra_custom_monitor_row(index):
                suffix = f"_{index}"
                return ui.div(
                    ui.hr(),
                    ui.input_text(f"scan_custom_name{suffix}", "Monitor name"),
                    ui.input_text(f"scan_custom_signature{suffix}",
                                  "Monitor signature (comma separated)",
                                  placeholder="timestamp, temperature, humidity"),
                    ui.input_text(f"scan_custom_time_column{suffix}", "Monitor timestamp column"),
                    ui.input_checkbox(f"scan_custom_use_format{suffix}",
                                      "Specify monitor custom timestamp format", value=False),
                    ui.panel_conditional(
                        f"input.scan_custom_use_format{suffix}",
                        ui.input_text(f"scan_custom_date_format{suffix}",
                                      "Custom timestamp format",
                                      placeholder="%Y-%m-%d %H:%M:%S"),
                    ),
                    class_="summary-custom-monitor-entry",
                )

            @reactive.effect
            def append_custom_monitor_editor():
                """Append one blank editor when the last row has user content.

                insert_ui appends without rebuilding preceding rows, retaining
                input values and focus across additions.
                """
                if "create" not in blank_tabs.get() or generated_summary.get() is not None:
                    return
                index = scan_custom_count.get() - 1
                name = scan_custom_field(index, "name")
                signature = scan_custom_field(index, "signature")
                time_column = scan_custom_field(index, "time_column")
                use_format = scan_custom_field(index, "use_format")
                if not any((str(value or "").strip() for value in
                            (name, signature, time_column))) and not use_format:
                    return
                # Don't append another entry until the latest row has been
                # mounted. Counts are session-local and reset on tab close.
                scan_custom_count.set(index + 2)
                ui.insert_ui(extra_custom_monitor_row(index + 1),
                             selector="#scan-custom-monitor-entries", where="beforeEnd")

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
                         and all(levels) and not scan_busy.get()
                         and not summary_busy.get() and not upas_filter_busy.get())
                return ui.input_action_button("scan_start", "Start scanning",
                                              class_="btn-primary w-100" if ready else "btn-outline-secondary w-100",
                                              disabled=not ready)

            @reactive.effect
            @reactive.event(input.scan_start)
            async def start_scan():
                nonlocal scanned_dataset
                if (dataset.get() is not None or scan_busy.get() or summary_busy.get()
                        or upas_filter_busy.get() or "create" not in blank_tabs.get()):
                    return
                epoch = create_epoch.get()
                levels = current_scan_levels()
                try:
                    # Fail quickly before scheduling the potentially slow scan.
                    params = scan_parameters(scan_folder_path.get(), levels,
                                             input.scan_timezone(), scan_metadata.get(),
                                             signatures=current_custom_monitor_signatures())
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
                    if params.get("signatures"):
                        scan_options["signatures"] = params["signatures"]
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
            def upas_filter_controls():
                """Filter-summary operations act on the scanned Dataset, not Summary."""
                ready = (scan_completed.get() and scanned_dataset is not None
                         and generated_summary.get() is None
                         and not scan_busy.get() and not summary_busy.get()
                         and not upas_filter_busy.get())
                return ui.div(
                    ui.h6("UPAS filter summary"),
                    ui.input_action_button("save_upas_filter_summary", "Save upas filter summary",
                                           disabled=not ready, class_="btn-outline-primary w-100 mb-2"),
                    ui.input_action_button("load_upas_filter_summary", "Load upas filter summary",
                                           disabled=not ready, class_="btn-outline-primary w-100"),
                    ui.p("Use these controls after scanning UPAS monitors and before creating the Summary.",
                         class_="form-text"),
                )

            @reactive.effect
            @reactive.event(input.save_upas_filter_summary)
            def open_save_upas_filter_dialog():
                if scanned_dataset is None or not scan_completed.get() or upas_filter_busy.get():
                    return
                folder = scan_folder_path.get() or str(Path.home())
                ui.modal_show(ui.modal(
                    ui.p("Download the current filter summary as filter_summary.csv. "
                         "Your browser will choose where to save it according to its download settings."),
                    ui.download_button("upas_save_download", "Save filter_summary.csv",
                                       class_="btn-primary"),
                    title="Save upas filter summary", easy_close=True,
                    footer=ui.modal_button("Cancel"),
                ))

            @reactive.effect
            @reactive.event(input.load_upas_filter_summary)
            def open_load_upas_filter_dialog():
                if scanned_dataset is None or not scan_completed.get() or upas_filter_busy.get():
                    return
                ui.modal_show(ui.modal(
                    ui.input_file("upas_load_upload", "Choose filter summary CSV",
                                  accept=[".csv"], multiple=False),
                    ui.p("Select a CSV file from your computer. Its values will be applied "
                         "to the currently scanned Dataset.", class_="form-text"),
                    title="Load upas filter summary", easy_close=True,
                    footer=ui.modal_button("Cancel"),
                ))

            @render.download(filename="filter_summary.csv")
            def upas_save_download():
                """Generate the existing Dataset CSV, then serve it to the browser."""
                from .dataset_creation import save_upas_filter_summary_file
                if (scanned_dataset is None or not scan_completed.get()
                        or generated_summary.get() is not None or scan_busy.get()
                        or summary_busy.get() or upas_filter_busy.get()):
                    raise ValueError("Scan UPAS monitors before saving their filter summary")
                epoch = create_epoch.get()
                upas_filter_busy.set(True)
                try:
                    with tempfile.TemporaryDirectory(prefix="apmtools-upas-") as directory:
                        target = save_upas_filter_summary_file(scanned_dataset, directory)
                        content = target.read_bytes()
                    if epoch == create_epoch.get():
                        append_operation_messages(["UPAS filter summary prepared for download: filter_summary.csv"], epoch)
                        status.set("UPAS filter summary prepared for download")
                    yield content
                except Exception as exc:
                    if epoch == create_epoch.get():
                        append_operation_messages([f"UPAS filter summary save failed: {exc}"], epoch)
                        ui.notification_show(f"Could not save UPAS filter summary: {exc}",
                                             type="error", duration=10)
                    raise
                finally:
                    if epoch == create_epoch.get():
                        upas_filter_busy.set(False)

            @reactive.effect
            @reactive.event(input.upas_load_upload)
            def run_load_upas_filter_summary():
                from .dataset_creation import load_upas_filter_summary_file
                if (scanned_dataset is None or not scan_completed.get()
                        or generated_summary.get() is not None or scan_busy.get()
                        or summary_busy.get() or upas_filter_busy.get()):
                    return
                selected = input.upas_load_upload()
                if not selected:
                    return
                epoch = create_epoch.get()
                upas_filter_busy.set(True)
                try:
                    upload = selected[0]
                    if Path(upload["name"]).suffix.lower() != ".csv":
                        raise ValueError("Select a .csv filter summary file")
                    # Shiny stores the uploaded CSV in a temporary server-side file.
                    # It may lack a .csv extension; retain its name for validation.
                    with tempfile.TemporaryDirectory(prefix="apmtools-upas-load-") as directory:
                        copied = Path(directory) / "filter_summary.csv"
                        import shutil
                        shutil.copyfile(upload["datapath"], copied)
                        source = load_upas_filter_summary_file(scanned_dataset, copied)
                    if epoch != create_epoch.get():
                        return
                    scan_output.set(scan_results_frame(scanned_dataset))
                    append_operation_messages([f"UPAS filter summary loaded: {upload['name']}"], epoch)
                    status.set(f"UPAS filter summary loaded: {upload['name']}")
                    ui.modal_remove()
                    ui.notification_show(f"UPAS filter summary loaded: {upload['name']}",
                                         type="message", duration=8)
                except Exception as exc:
                    if epoch == create_epoch.get():
                        append_operation_messages([f"UPAS filter summary load failed: {exc}"], epoch)
                        ui.notification_show(f"Could not load UPAS filter summary: {exc}",
                                             type="error", duration=10)
                finally:
                    if epoch == create_epoch.get():
                        upas_filter_busy.set(False)

            def _modify_source():
                source = dataset.get()
                return source if source is not None else generated_summary.get()

            def _active_modify_timeline():
                source = _modify_source()
                if source is None:
                    return None
                timeline = modify_timeline.get()
                if timeline is None or timeline.current is not source:
                    from .summary_timeline import SummaryStepTimeline
                    timeline = SummaryStepTimeline(source)
                    modify_timeline.set(timeline)
                return timeline

            @render.ui
            def modify_result_content():
                """Preview the current Summary without copying the full dataset."""
                summary = dataset.get()
                if summary is None:
                    summary = generated_summary.get()
                if summary is None:
                    return ui.div(ui.p("Create or open a Summary to modify it."), class_="p-3")
                return ui.div(
                    ui.div(
                        ui.strong(f"Summary rows: {len(summary):,}"),
                        ui.input_select("modify_preview_rows", "Rows in grid preview",
                                        choices={"500": "500 rows", "1000": "1,000 rows",
                                                 "5000": "5,000 rows", "all": "All rows"},
                                        selected="1000"),
                        class_="d-flex align-items-center justify-content-between gap-3 flex-wrap summary-scan-preview-toolbar",
                    ),
                    ui.div(ui.output_data_frame("modify_results_grid"), class_="summary-scan-grid"),
                    class_="summary-scan-results-content",
                )

            @render.data_frame
            def modify_results_grid():
                from shiny import render as shiny_render
                import pandas as pd
                summary = dataset.get()
                if summary is None:
                    summary = generated_summary.get()
                if summary is None:
                    return shiny_render.DataGrid(pd.DataFrame(), width="100%")
                try:
                    row_count = input.modify_preview_rows() or "1000"
                except Exception:
                    row_count = "1000"
                preview = _table_preview_frame(summary, row_count).copy()
                for name in preview.columns:
                    if preview[name].dtype == object:
                        preview[name] = preview[name].map(
                            lambda value: value if isinstance(value, (str, int, float, bool)) or value is None
                            else str(value))
                return shiny_render.DataGrid(preview, filters=True, width="100%", height="100%")

            @render.ui
            def modify_editor_sidebar():
                """Structural column edits in the left Modify dataset sidebar."""
                summary = _modify_source()
                # A committed edit reconstructs this sidebar. The previously
                # inserted condition inputs disappear, so reset their count.
                if summary is not modify_map_source_ref[0]:
                    modify_map_source_ref[0] = summary
                    if modify_map_condition_count.get() != 1:
                        modify_map_condition_count.set(1)
                if summary is None:
                    return ui.p("Open or create a Summary first.")
                if not len(summary.columns):
                    return ui.p("No columns available")
                import pandas as pd
                from .shiny.datetime_intervals import MAX_DATETIME_INTERVALS
                columns = list(map(str, summary.columns))
                numeric_columns = [c for c in columns if pd.api.types.is_numeric_dtype(summary[c])]
                filter_stages = min(MAX_DATETIME_INTERVALS, max(2, len(columns) + 1))
                subset_fields = []
                for stage in range(filter_stages):
                    control = ui.div(
                        ui.input_select(f"modify_filter_column_{stage}", f"Subset column {stage + 1}",
                            choices={"": "Select a column", **{c: c for c in columns}}),
                        ui.output_ui(f"modify_filter_values_ui_{stage}"),
                        ui.input_select(f"modify_filter_style_{stage}", "Filter style",
                            choices={k: k for k in ("all", "any", "negative all", "negative any")}),
                        class_="mb-3")
                    if stage:
                        control = ui.panel_conditional(
                            " && ".join(f"input.modify_filter_column_{i} !== ''" for i in range(stage)), control)
                    subset_fields.append(control)
                datetime_fields = []
                for stage in range(MAX_DATETIME_INTERVALS):
                    start_id = f"modify_filter_datetime_start_{stage}"
                    end_id = f"modify_filter_datetime_end_{stage}"
                    control = ui.div(
                        ui.tags.strong(f"Datetime interval {stage + 1}"),
                        ui.input_select(f"modify_filter_datetime_mode_{stage}", "Filter type",
                            choices={"positive": "Positive (keep)", "negative": "Negative (exclude)"}),
                        ui.input_text(start_id, "Start datetime",
                            placeholder="YYYY-MM-DD, HH:MM:SS, or YYYY-MM-DD HH:MM:SS"),
                        ui.input_text(end_id, "End datetime",
                            placeholder="YYYY-MM-DD, HH:MM:SS, or YYYY-MM-DD HH:MM:SS"),
                        class_="mb-3")
                    if stage:
                        prev = stage - 1
                        control = ui.panel_conditional(
                            f"(input.modify_filter_datetime_start_{prev} && input.modify_filter_datetime_start_{prev}.trim() !== '') || "
                            f"(input.modify_filter_datetime_end_{prev} && input.modify_filter_datetime_end_{prev}.trim() !== '')",
                            control)
                    datetime_fields.append(control)
                return ui.navset_tab(
                    ui.nav_panel("Edit columns",
                        ui.h6("Edit column names"),
                        ui.input_text_area("modify_rename_mapping", "Old=new (one per line)", rows=5,
                                           value=editing.rename_defaults(summary)),
                        ui.input_action_button("modify_apply_rename", "Apply names", class_="btn-outline-primary w-100"),
                        ui.hr(),
                        ui.h6("Keep columns"),
                        ui.input_text_area("modify_keep_columns", "Comma-separated names to keep", rows=3,
                                           value=editing.keep_defaults(summary)),
                        ui.input_action_button("modify_apply_keep", "Keep listed columns", class_="btn-outline-primary w-100"),
                        ui.hr(),
                        ui.h6("Create column"),
                        ui.input_text("modify_new_name", "Column name"),
                        ui.input_text("modify_new_fill", "Initial value (optional)"),
                        ui.input_action_button("modify_apply_create", "Create column", class_="btn-outline-primary w-100"),
                    ),
                    ui.nav_panel("Filter dataset",
                        ui.h6("Subset"),
                        *subset_fields,
                        ui.hr(),
                        ui.input_selectize("modify_filter_numeric_columns", "Numerical subset",
                            choices={c: c for c in numeric_columns}, multiple=True),
                        ui.output_ui("modify_filter_numeric_bounds"),
                        ui.hr(),
                        ui.h6("Datetime filters"),
                        *datetime_fields,
                        ui.help_text("Positive intervals are combined (keep any matching interval); negative intervals exclude matching observations."),
                        ui.input_action_button("modify_apply_filter", "Apply filters", class_="btn-outline-primary w-100"),
                    ),
                    id="modify_editor_tabs",
                )

            from .shiny.data import _subset_choices
            from .shiny.datetime_intervals import MAX_DATETIME_INTERVALS, normalize_datetime_intervals

            def _modify_filter_values_ui(stage):
                summary = _modify_source()
                if summary is None:
                    return ui.div()
                try:
                    column = input[f"modify_filter_column_{stage}"]()
                except Exception:
                    return ui.div()
                if column not in summary.columns:
                    return ui.div()
                choices, _ = _subset_choices(summary, column)
                return ui.input_selectize(f"modify_filter_values_{stage}", f"{column} values",
                    choices=choices, multiple=True)

            for _filter_stage in range(MAX_DATETIME_INTERVALS):
                def _make_filter_values(stage):
                    def result():
                        return _modify_filter_values_ui(stage)
                    result.__name__ = f"modify_filter_values_ui_{stage}"
                    renderer = render.ui(result)
                    if output is not None:
                        output(renderer)
                    return renderer
                _make_filter_values(_filter_stage)

            @render.ui
            def modify_filter_numeric_bounds():
                summary = _modify_source()
                if summary is None:
                    return ui.div()
                try:
                    selected = list(input.modify_filter_numeric_columns() or [])
                except Exception:
                    selected = []
                fields = []
                for column in selected:
                    if column not in summary.columns:
                        continue
                    fields.append(ui.div(
                        ui.tags.strong(column),
                        ui.input_text(f"modify_filter_numeric_gt_{list(summary.columns).index(column)}", "Greater than"),
                        ui.input_text(f"modify_filter_numeric_lt_{list(summary.columns).index(column)}", "Less than"),
                        class_="mb-3"))
                return ui.div(*fields)

            def _read_modify_filter_spec():
                summary = _modify_source()
                if summary is None:
                    raise ValueError("No Summary loaded")
                stages = min(MAX_DATETIME_INTERVALS, max(2, len(summary.columns) + 1))
                chain = []
                for stage in range(stages):
                    column = input[f"modify_filter_column_{stage}"]()
                    if column not in summary.columns:
                        break
                    choices, mapping = _subset_choices(summary, column)
                    keys = list(input[f"modify_filter_values_{stage}"]() or [])
                    values = [mapping[key] for key in keys if key in mapping]
                    if values:
                        chain.append({"column": column, "values": values,
                            "style": input[f"modify_filter_style_{stage}"]() or "all"})
                intervals = []
                for stage in range(MAX_DATETIME_INTERVALS):
                    start = input[f"modify_filter_datetime_start_{stage}"]() or ''
                    end = input[f"modify_filter_datetime_end_{stage}"]() or ''
                    if not str(start).strip() and not str(end).strip():
                        break
                    intervals.append({"start": start, "end": end,
                        "type": input[f"modify_filter_datetime_mode_{stage}"]() or "positive"})
                bounds = {}
                for column in list(input.modify_filter_numeric_columns() or []):
                    if column not in summary.columns:
                        continue
                    index = list(summary.columns).index(column)
                    selected_bounds = {}
                    for key in ('gt', 'lt'):
                        raw = input[f"modify_filter_numeric_{key}_{index}"]()
                        if str(raw or '').strip():
                            selected_bounds[key] = raw.strip()
                    if selected_bounds:
                        bounds[column] = selected_bounds
                return chain, normalize_datetime_intervals(intervals), bounds

            @reactive.effect
            @reactive.event(input.modify_apply_filter)
            def modify_apply_filter():
                try:
                    chain, intervals, bounds = _read_modify_filter_spec()
                except Exception as exc:
                    ui.notification_show(f"Invalid dataset filters: {exc}", type="error", duration=10)
                    return
                if not (chain or intervals or bounds):
                    ui.notification_show("Select at least one filter", type="warning")
                    return
                _run_modify(lambda data: data.filter_dataset(chain, intervals, bounds))

            @render.ui
            def modify_editor_right_sidebar():
                """Value edits and history in the right Modify dataset sidebar."""
                summary = _modify_source()
                if summary is None:
                    return ui.p("Open or create a Summary first.")
                if not len(summary.columns):
                    return ui.p("No columns available")
                timeline = _active_modify_timeline()
                names = list(map(str, summary.columns))
                choices = {name: name for name in names}
                return ui.navset_tab(
                    ui.nav_panel("Edit/Map column values",
                        ui.h6("Edit column values"),
                        ui.input_select("modify_values_column", "Column", choices=choices),
                        ui.output_ui("modify_values_mapping_editor"),
                        ui.input_action_button("modify_apply_values", "Apply values", class_="btn-outline-primary w-100"),
                        ui.hr(),
                        ui.h6("Map values"),
                        ui.input_select("modify_map_target", "Target column", choices=choices),
                        ui.input_select("modify_map_condition", "Condition column", choices=choices),
                        ui.input_select("modify_map_operator", "Condition",
                                        choices={op: op for op in ('==', '!=', '>', '>=', '<', '<=', 'isin', 'isna', 'notna')}),
                        ui.input_text("modify_map_value", "Compare with (comma-separated for isin)"),
                        ui.div(id="modify-map-extra-conditions"),
                        ui.input_action_button("modify_map_add_condition", "+ Add condition", class_="btn-outline-secondary btn-sm"),
                        ui.p("Conditions are combined from top to bottom (left to right).", class_="form-text"),
                        ui.input_text("modify_map_true", "Value when True"),
                        ui.input_checkbox("modify_map_has_false", "Set value when False", value=False),
                        ui.input_text("modify_map_false", "Value when False (optional)"),
                        ui.input_action_button("modify_apply_map", "Apply mapping", class_="btn-outline-primary w-100"),
                        ui.hr(),
                        ui.h6("Calculate column"),
                        ui.input_text("modify_calc_name", "New or existing column name"),
                        ui.input_text_area("modify_calc_expression", "Expression", rows=3,
                                           placeholder="a * 0.7 + b * 0.3"),
                        ui.input_action_button("modify_apply_calculate", "Calculate column", class_="btn-outline-primary w-100"),
                    ),
                    ui.nav_panel("History",
                        ui.h6("Transformation history"),
                        ui.input_select("modify_history_step", "Dataset state",
                                        choices=timeline.options(), selected=str(timeline.cursor)),
                        ui.p("Select a step to preview that dataset state. Later steps remain available until you apply a new edit.", class_="form-text"),
                        ui.p(f"{timeline.history_prefix} earlier recorded step(s) were present when this Summary was opened. "
                             "Those intermediate states require the original source data to reconstruct."
                             if timeline.history_prefix else "Original editing state is the Summary when this editor opened.",
                             class_="form-text"),
                    ),
                    id="modify_editor_right_tabs",
                )

            @render.ui
            def modify_values_mapping_editor():
                summary = dataset.get()
                if summary is None:
                    summary = generated_summary.get()
                if summary is None or len(summary.columns) == 0:
                    return ui.div()
                try:
                    selected = input.modify_values_column()
                except Exception:
                    selected = None
                if selected not in summary.columns:
                    selected = str(summary.columns[0])
                values = editing.values_defaults(summary, selected)
                return ui.div(
                    ui.input_text_area("modify_values_mapping", "Old=new (one per line)",
                                       rows=5, value=values),
                    ui.p("Up to 200 distinct non-missing values are shown. Edit the right-hand side to replace values.",
                         class_="form-text"),
                )

            def _commit_modified_summary(result, *, message="Summary updated; change recorded in history."):

                source = dataset.get()
                if source is None:
                    self.created_summary = result
                    generated_summary.set(result)
                else:
                    # Refresh the plotting workspace with the edited Summary.
                    # Retain source-level plots; column transformations themselves
                    # clear stale views as defined by the Summary methods.
                    self.original_summary = result
                    self.summary = result.copy(deep=False)
                    dataset.set(self.summary)
                    # The loaded workspace owns a shallow Summary copy. Keep
                    # the timeline pointing at that exact live object.
                    timeline = modify_timeline.get()
                    if timeline is not None and timeline.current is result:
                        timeline.current = self.summary
                        modify_timeline.set(timeline)
                    generation.set(generation.get() + 1)
                status.set(f"Summary modified: {len(result):,} rows, {len(result.columns)} columns")
                ui.notification_show(message, type="message", duration=5)

            def _run_modify(operation):
                timeline = _active_modify_timeline()
                if timeline is None:
                    ui.notification_show("Open or create a Summary first", type="warning")
                    return
                try:
                    result = operation(timeline.current)
                    if result is timeline.current:
                        raise ValueError("Operation did not produce a new Summary")
                    timeline.append(result)
                    modify_timeline.set(timeline)
                    _commit_modified_summary(result)
                except Exception as exc:
                    ui.notification_show(f"Could not modify Summary: {exc}", type="error", duration=10)

            @reactive.effect
            @reactive.event(input.modify_history_step)
            def modify_select_history_step():
                timeline = _active_modify_timeline()
                if timeline is None:
                    return
                try:
                    index = int(input.modify_history_step())
                    if index == timeline.cursor:
                        return
                    result = timeline.select(index)
                    modify_timeline.set(timeline)
                    _commit_modified_summary(
                        result, message=f"Showing dataset at history step {index} of {timeline.count}")
                except Exception as exc:
                    ui.notification_show(f"Could not restore history step: {exc}",
                                         type="error", duration=10)

            @reactive.effect
            @reactive.event(input.modify_apply_rename)
            def modify_apply_rename():
                _run_modify(lambda data: editing.edit_rename(data, input.modify_rename_mapping()))

            @reactive.effect
            @reactive.event(input.modify_apply_values)
            def modify_apply_values():
                _run_modify(lambda data: editing.edit_values(data, input.modify_values_column(), input.modify_values_mapping()))

            @reactive.effect
            @reactive.event(input.modify_apply_keep)
            def modify_apply_keep():
                _run_modify(lambda data: editing.edit_keep_text(data, input.modify_keep_columns()))

            @reactive.effect
            @reactive.event(input.modify_apply_create)
            def modify_apply_create():
                _run_modify(lambda data: editing.create_column(data, input.modify_new_name(), input.modify_new_fill()))

            @reactive.effect
            @reactive.event(input.modify_map_add_condition)
            def modify_map_add_condition():
                """Append rather than rerender existing inputs to preserve edited values."""
                index = modify_map_condition_count.get()
                modify_map_condition_count.set(index + 1)
                summary = _modify_source()
                if summary is None:
                    return
                choices = {str(name): str(name) for name in summary.columns}
                operators = {op: op for op in ('==', '!=', '>', '>=', '<', '<=', 'isin', 'isna', 'notna')}
                ui.insert_ui(ui.div(
                    ui.hr(),
                    ui.input_select(f"modify_map_join_{index}", "Combine with previous", choices={'and': 'AND', 'or': 'OR'}),
                    ui.input_select(f"modify_map_condition_{index}", "Condition column", choices=choices),
                    ui.input_select(f"modify_map_operator_{index}", "Condition", choices=operators),
                    ui.input_text(f"modify_map_value_{index}", "Compare with (comma-separated for isin)"),
                    class_="summary-map-condition-entry"),
                    selector="#modify-map-extra-conditions", where="beforeEnd")

            @reactive.effect
            @reactive.event(input.modify_apply_map)
            def modify_apply_map():
                conditions = [{'column': input.modify_map_condition(),
                               'op': input.modify_map_operator(),
                               'value': input.modify_map_value()}]
                for index in range(1, modify_map_condition_count.get()):
                    conditions.append({
                        'join': input[f"modify_map_join_{index}"](),
                        'column': input[f"modify_map_condition_{index}"](),
                        'op': input[f"modify_map_operator_{index}"](),
                        'value': input[f"modify_map_value_{index}"]()})
                _run_modify(lambda data: editing.map_column_conditions(
                    data, input.modify_map_target(), conditions,
                    input.modify_map_true(), input.modify_map_false(),
                    bool(input.modify_map_has_false())))

            @reactive.effect
            @reactive.event(input.modify_apply_calculate)
            def modify_apply_calculate():
                _run_modify(lambda data: editing.calculate_column(data, input.modify_calc_name(), input.modify_calc_expression()))

            def modify_workspace():
                """Editable Summary with edge-collapse controls and draggable sidebars."""
                def side_panel(side, output):
                    return ui.div(
                        ui.div(ui.output_ui(output), class_="summary-modify-sidebar-body"),
                        ui.tags.button("‹" if side == "left" else "›", type="button",
                            class_="summary-modify-edge-toggle", **{
                                "data-side": side, "aria-label": f"Collapse {side} sidebar",
                                "aria-expanded": "true", "title": f"Collapse {side} sidebar"}),
                        ui.tags.div(class_="summary-modify-resize-grip",
                            **{"data-side":side, "role":"separator",
                               "aria-label":f"Drag to resize {side} sidebar",
                               "title":f"Drag to resize {side} sidebar"}),
                        class_=f"summary-modify-{side}",
                    )
                return ui.div(
                    ui.div(
                        side_panel("left", "modify_editor_sidebar"),
                        ui.div(ui.output_ui("modify_result_content"),
                               class_="summary-empty-dataset-main summary-create-main summary-modify-main"),
                        side_panel("right", "modify_editor_right_sidebar"),
                        class_="summary-modify-panels",
                    ),
                    class_="summary-modify-workspace",
                )

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
                if dataset.get() is None and generated_summary.get() is None:
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
                nonlocal scanned_dataset
                if dataset.get() is None and generated_summary.get() is None:
                    return
                self.original_summary = None
                self.summary = None
                dataset.set(None)
                modify_timeline.set(None)
                modify_open.set(False)
                selected_loaded_tab.set("plots")
                # Close also clears an in-memory Summary created in the wizard.
                self.created_summary = None
                generated_summary.set(None)
                scan_completed.set(False)
                scan_output.set(None)
                scanned_dataset = None
                create_epoch.set(create_epoch.get() + 1)
                plot_open.set(False)
                generation.set(generation.get() + 1)
                blank_tabs.set(())
                selected_blank_tab.set(None)
                status.set("No dataset loaded")
                ui.modal_remove()

            @reactive.effect
            @reactive.event(input.save_dataset_root)
            def show_root_save_dialog():
                if dataset.get() is None and generated_summary.get() is None:
                    ui.notification_show("Open or create a dataset before saving", type="warning")
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
                        (self.original_summary if dataset.get() is not None else generated_summary.get()),
                        input.root_dataset_path(),
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
                modify_timeline.set(None)
                modify_open.set(False)
                selected_loaded_tab.set("plots")
                plot_open.set(False)
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
                                ui.input_action_button(
                                    "close_create_dataset_tab" if mode == "create" else "close_modify_dataset_tab",
                                    "×", class_="btn btn-sm btn-link text-danger p-0 ms-2",
                                    title=f"Close {labels[mode]} tab"),
                                class_="d-inline-flex align-items-center")),
                            (ui.div(
                                (ui.div(
                                    ui.navset_tab(
                                        *( [ui.nav_panel(ui.tags.span("Scan folder", class_="summary-scan-tab-label"),
                                            ui.output_ui("scan_folder_button"),
                                            ui.output_ui("scan_levels_ui"),
                                            ui.input_numeric("scan_timezone", "UTC timezone", value=0, step=0.5),
                                            ui.input_action_button("scan_add_metadata_dialog", "Add metadata",
                                                                   class_="btn-outline-secondary w-100"),
                                            ui.output_ui("scan_metadata_list"),
                                            ui.hr(),
                                            ui.h5("Scan custom monitors"),
                                            ui.input_text("scan_custom_name", "Monitor name"),
                                            ui.input_text("scan_custom_signature", "Monitor signature (comma separated)",
                                                          placeholder="timestamp, temperature, humidity"),
                                            ui.input_text("scan_custom_time_column", "Monitor timestamp column"),
                                            ui.input_checkbox("scan_custom_use_format", "Specify monitor custom timestamp format", value=False),
                                            ui.output_ui("scan_custom_timestamp_format"),
                                            ui.div(id="scan-custom-monitor-entries"),
                                            ui.output_ui("scan_start_control"),
                                            ui.hr(),
                                            ui.output_ui("upas_filter_controls"),
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
                                class_="summary-empty-dataset-panels summary-create-panels",
                            ) if mode == "create" else modify_workspace()),
                            value=mode,
                        ) for mode in tabs),
                        id="blank_dataset_tabs",
                        selected=selected_blank_tab.get(),
                    )
                if not plot_open.get():
                    if modify_open.get():
                        return modify_workspace()
                    return ui.div(
                        ui.h3("Summary ready"),
                        ui.p("Choose Modify Dataset or Plot/Table Dataset from the toolbar."),
                        class_="p-4",
                    )
                name = f"dataset_{generation.get()}"
                # Mount effect is responsible for module registration.
                # This render function only creates the UI markup.
                if mounted_generation.get() != generation.get():
                    return ui.div("Loading plotting workspace...")
                workspace = ui_factories[name](name)
                if not modify_open.get():
                    return workspace
                return ui.navset_tab(
                    ui.nav_panel("Plots", workspace, value="plots"),
                    ui.nav_panel(
                        ui.tags.span("Modify dataset",
                            ui.input_action_button("close_modify_dataset_tab", "×",
                                class_="btn btn-sm btn-link text-danger p-0 ms-2",
                                title="Close Modify dataset tab"),
                            class_="d-inline-flex align-items-center"),
                        modify_workspace(), value="modify"),
                    id="loaded_dataset_tabs", selected=selected_loaded_tab.get(),
                )

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

# SimWork-UI
SimWork-UI 

A minimal overview of the project and its top-level layout.

SimWork-UI/
├─ agent/
│  ├─ agent_controller.py
│  ├─ execution_context.py
│  ├─ introspection_logger.py
│  ├─ logging_agent.py
│  └─ policy_base.py
├─ dataset/
│  ├─ clip_index.py
│  ├─ dd7dfa19-6502-46d5-9e5b-69f01412ddb3.clips.json
│  ├─ dd7dfa19-6502-46d5-9e5b-69f01412ddb3.npz
│  ├─ recorder.py
│  ├─ session_store.py
│  ├─ session_types.py
│  ├─ trace_store.py
│  └─ views.py
├─ main.py
├─ runner/
│  ├─ eval_runner.py
│  ├─ live_runner.py
│  └─ playback_runner.py
├─ utils/
│  ├─ config.py
│  ├─ time_series.py
│  └─ validation.py
└─ viewer/
   ├─ app_controller.py
   ├─ event_bus.py
   ├─ live_env_panel.py
   ├─ snapshot_panel.py
   ├─ trace_browser_panel.py
   └─ trace_player_panel.py

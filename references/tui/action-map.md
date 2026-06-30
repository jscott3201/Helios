# TUI Action Map

## Global actions

| Key | Action | Destructive | Notes |
|---|---|---:|---|
| `q` | quit/back | no | In modal, closes modal first. |
| `Ctrl-C` | cancel generation or quit confirmation | maybe | If generation active, cancel request. |
| `?` | help overlay | no | Shows current-scope keys. |
| `1` | dashboard | no | |
| `2` | chat | no | |
| `3` | adapters | no | |
| `4` | cache | no | |
| `5` | benchmarks | no | |
| `6` | config | no | |
| `7` | logs | no | |
| `Tab` | next focus | no | |
| `Shift-Tab` | previous focus | no | |
| `/` | filter/search | no | Current table/log. |
| `Ctrl-R` | refresh | no | Force snapshot refresh. |

## Destructive confirmations

| Action | Confirmation text |
|---|---|
| cache eviction | `Type EVICT to remove selected cache blocks.` |
| adapter unload | `Type UNLOAD to unload selected adapter.` |
| config apply requiring restart | `Type APPLY to write config; restart still required.` |
| server stop | `Type STOP to stop local server.` |

The UI must not map destructive actions to a single accidental keypress.

# TUI Keybindings

## Global

| Key | Action |
|---|---|
| `Tab` / `Shift-Tab` | Next / previous page |
| `1`..`9` | Direct page jump |
| `?` | Help |
| `:` | Command palette |
| `/` | Search/filter current page |
| `r` | Refresh current page |
| `q` | Quit or close current modal/context |
| `Esc` | Close modal/cancel selection |
| `Ctrl-C` | Graceful shutdown |

## Benchmarks

| Key | Action |
|---|---|
| `b` | Run selected benchmark |
| `s` | Stop active benchmark |
| `c` | Copy exact command/output path |
| `e` | Export reproduction bundle |

## Config

| Key | Action |
|---|---|
| `v` | Validate current config |
| `d` | Show diff before save |
| `w` | Write after confirmation |

## Safety rule

Every destructive operation must present a confirmation modal. The reducer must make confirmation state explicit and testable.

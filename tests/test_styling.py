import io

from rich.console import Console

import inference


def make_record_console() -> Console:
    """Console that records all output to a string buffer, no ANSI, no terminal."""
    return Console(file=io.StringIO(), force_terminal=False, record=True, width=80)


def test_live_markdown_out_is_context_manager():
    console = make_record_console()
    with inference._LiveMarkdownOut(console) as live_out:
        live_out.write("hello")
    # After exit, the live region is torn down — no exception.


def test_live_markdown_out_accumulates_writes():
    console = make_record_console()
    with inference._LiveMarkdownOut(console) as live_out:
        live_out.write("hello")
        live_out.write(" world")
    # Final captured export contains the fully assembled text.
    assert "hello world" in console.export_text()


def test_live_markdown_out_flush_is_callable():
    console = make_record_console()
    with inference._LiveMarkdownOut(console) as live_out:
        live_out.write("x")
        live_out.flush()


def test_live_markdown_out_write_returns_length():
    console = make_record_console()
    with inference._LiveMarkdownOut(console) as live_out:
        n = live_out.write("abc")
    assert n == 3


def test_live_markdown_out_does_nothing_when_no_writes():
    console = make_record_console()
    with inference._LiveMarkdownOut(console):
        pass
    # No panic, no output — exit must handle the never-started case.
    assert console.export_text() == ""


def test_welcome_banner_contains_model_name():
    console = make_record_console()
    inference._welcome_banner(console, "my-model", disable_thinking=False)
    out = console.export_text()
    assert "my-model" in out
    assert "inference" in out  # panel title
    assert "thinking: disabled" not in out  # omitted when flag off


def test_welcome_banner_shows_thinking_disabled_when_on():
    console = make_record_console()
    inference._welcome_banner(console, "my-model", disable_thinking=True)
    out = console.export_text()
    assert "thinking: disabled" in out


def test_success_prints_green_check():
    console = make_record_console()
    inference._success(console, "History cleared.")
    out = console.export_text()
    assert "✓ History cleared." in out


def test_error_prints_bracketed_prefix():
    console = make_record_console()
    inference._error(console, "boom")
    out = console.export_text()
    assert "[error] boom" in out


def test_cancelled_prints_yellow_bracket():
    console = make_record_console()
    inference._cancelled(console)
    out = console.export_text()
    assert "[cancelled]" in out


def test_interrupted_prints_yellow_bracket():
    console = make_record_console()
    inference._interrupted(console)
    out = console.export_text()
    assert "[interrupted]" in out


def test_info_prints_text():
    console = make_record_console()
    inference._info(console, "Switched to m.")
    out = console.export_text()
    assert "Switched to m." in out


def test_help_line_prints_command_list():
    console = make_record_console()
    inference._help_line(console)
    out = console.export_text()
    assert "/clear" in out
    assert "/model" in out
    assert "/system" in out
    assert "/add" in out
    assert "/remove" in out
    assert "/exit" in out

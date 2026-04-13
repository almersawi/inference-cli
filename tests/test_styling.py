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

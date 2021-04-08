import curses

def c_main(stdscr: "curses._CursesWindow"):
    stdscr.clear()
    name = ""
    name_done = False
    while True:
        stdscr.addstr(0, 0, "what is your name? ")
        stdscr.clrtoeol()
        stdscr.addstr(name)
        if name_done:
            stdscr.addstr(1, 0, f"Hello {name}")

        char = stdscr.get_wch()
        if isinstance(char, str) and char.isprintable():
            name += char
        elif char == curses.KEY_BACKSPACE:
            name = name[:-1]
        elif char == "\n":
            if name_done:
                return 1
            name_done = True
        else:
            raise AssertionError(repr(char))
    return 0

def window() -> int:
    return curses.wrapper(c_main)

import curses

def c_main(stdscr: "curses._CursesWindow"):
    stdscr.clear()
    name = ""
    name_done = False
    init_pos = len("what is your name? ")
    x = init_pos
    while True:
        stdscr.addstr(0, 0, "what is your name? ")
        stdscr.clrtoeol()
        stdscr.addstr(name)
        if name_done:
            stdscr.addstr(1, 0, f"Hello {name}")
        stdscr.addstr(0, x, "")

        char = stdscr.get_wch()
        if isinstance(char, str) and char.isprintable():
            name += char
            x += 1
        elif char == curses.KEY_BACKSPACE or char == "\x7f":
            name = name[:-1]
            x -= 1
        elif char == "\n":
            if name_done:
                return 1
            name_done = True
        elif char == curses.KEY_LEFT:
            x -= 1
        elif char == curses.KEY_RIGHT:
            x += 1
        else:
            raise AssertionError(repr(char))
    return 0

def window() -> int:
    return curses.wrapper(c_main)

window()

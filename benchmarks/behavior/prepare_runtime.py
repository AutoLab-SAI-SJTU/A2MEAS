"""Apply the two lifecycle fixes to the pinned Isaac Sim 5.1 extensions."""
import fcntl
from pathlib import Path
import sysconfig


PATCHES = {
    'exts/isaacsim.simulation_app/isaacsim/simulation_app/simulation_app.py': [
        ('            self._app.print_and_log("Simulation App Shutting Down")',
         '            carb.log_info("Simulation App Shutting Down")'),
        ('        self._app.print_and_log("Simulation App Startup Complete")',
         '        carb.log_info("Simulation App Startup Complete")'),
        ('        print("Starting kit application with the following args: ", args)',
         '        carb.log_info(f"Starting kit application with the following args: {args}")'),
        ('        print("Passing the following args to the base kit application: ", unknown_args)',
         '        carb.log_info(f"Passing the following args to the base kit application: {unknown_args}")'),
    ],
    'extscache/omni.kit.scripting-107.3.2+107.3.2/omni/kit/scripting/scripts/loader/omni_cache.py': [
        ('import shutil\n', 'import shutil\nimport tempfile\n'),
        ('''        cls._path = Path.home() / ".nvidia-omniverse" / "pycache"
        try:
            if cls._path.exists():
                shutil.rmtree(str(cls._path))
            cls._path.mkdir(parents=True)
        except:
            carb.log_info("Python Scripting: skipping OmniCache cleanup.")

        if not cls._path.exists():
            carb.log_error("Python Scripting: Failed to access OmniCache directory.")
''', '''        # Each process owns its script cache; never delete another worker's cache.
        cls._path = Path(tempfile.mkdtemp(prefix="omni-scripting-"))
'''),
    ],
    'extscache/omni.kit.window.property-1.12.1+69cbf6ad/omni/kit/window/property/window.py': [
        ('''        # scroll position is at the bottom of the window
        if int(self.properties_frame.scroll_y_max) == int(self.properties_frame.scroll_y):''',
         '''        frame = self.properties_frame
        if frame is None:
            return
        # scroll position is at the bottom of the window
        if int(frame.scroll_y_max) == int(frame.scroll_y):'''),
        ('        self.__old_scroll_pos = self.properties_frame.scroll_y\n',
         '        self.__old_scroll_pos = frame.scroll_y\n'),
        ('''            # scroll position is at the bottom of the window
            if ypos == sys.float_info.max:
                self.properties_frame.scroll_y = self.properties_frame.scroll_y_max
            else:
                self.properties_frame.scroll_y = ypos''',
         '''            frame = self.properties_frame
            if frame is None:
                return
            # scroll position is at the bottom of the window
            if ypos == sys.float_info.max:
                frame.scroll_y = frame.scroll_y_max
            else:
                frame.scroll_y = ypos'''),
    ],
}


def prepare(root):
    root = Path(root)
    with (root / '.a2meas-patches.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        updates = []
        for name, replacements in PATCHES.items():
            path = root / name
            original = path.read_bytes()
            newline = b'\r\n' if b'\r\n' in original else b'\n'
            source = original.decode().replace('\r\n', '\n')
            for old, new in replacements:
                if new in source:
                    continue
                if source.count(old) != 1:
                    raise ValueError(f'Unsupported Isaac Sim extension source: {path}')
                source = source.replace(old, new, 1)
            updated = source.encode().replace(b'\n', newline)
            if updated != original:
                updates.append((path, updated))
        for path, content in updates:
            temporary = path.with_suffix('.a2meas.tmp')
            try:
                temporary.write_bytes(content)
                temporary.chmod(path.stat().st_mode & 0o777)
                temporary.replace(path)
            finally:
                temporary.unlink(missing_ok=True)


if __name__ == '__main__':
    prepare(Path(sysconfig.get_path('purelib')) / 'isaacsim')

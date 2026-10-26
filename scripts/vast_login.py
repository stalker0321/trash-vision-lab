"""Save a Vast API key via a hidden local-terminal prompt, outside the repo."""
import getpass
import os
from pathlib import Path
import sys
import tempfile


def save_key(key, destination):
    key = key.strip()
    if not key or any(c.isspace() or ord(c) < 32 for c in key):
        raise ValueError('Ключ пустой или содержит пробельные/управляющие символы.')
    destination.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    fd, temporary = tempfile.mkstemp(prefix='.api-key-', dir=destination.parent)
    try:
        with os.fdopen(fd, 'w', encoding='utf-8') as stream:
            os.fchmod(stream.fileno(), 0o600)
            stream.write(key + '\n')
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, destination)
    finally:
        Path(temporary).unlink(missing_ok=True)


def main():
    if not sys.stdin.isatty():
        raise SystemExit('Запусти этот скрипт в своём терминале: нужен скрытый ввод ключа.')
    destination = Path(os.environ.get('XDG_CONFIG_HOME') or Path.home() / '.config') / 'vastai/vast_api_key'
    print('API-ключ: https://cloud.vast.ai/manage-keys/?tab=api-keys')
    print('Ввод скрыт. Ключ сохранится локально, вне репозитория и истории команд.')
    if destination.exists():
        print('Существующий локальный ключ будет заменён введённым ключом.')
    try:
        key = getpass.getpass('Vast API key: ')
        save_key(key, destination)
    except (EOFError, KeyboardInterrupt):
        raise SystemExit('\nОтменено; ключ не сохранён.')
    except ValueError as error:
        raise SystemExit(str(error))
    print(f'Ключ сохранён: {destination} (доступ только владельцу).')
    print('Авторизация ещё не проверена; сообщи в чат, что ключ сохранён.')


if __name__ == '__main__':
    main()

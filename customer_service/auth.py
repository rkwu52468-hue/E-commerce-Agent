"""Local classroom credentials; no secrets or login tools are sent to the model.

This is an offline account verifier for synthetic customers, not a web identity
provider. Local maintenance provisions credentials; the live CLI verifies them.
"""
import getpass
import hashlib
import hmac
import json
import os
from pathlib import Path
import re
import secrets
import warnings


class CredentialError(ValueError):
    """A safe fixed message; never include supplied credentials or file contents."""


def read_secret(prompt):
    """Refuse getpass's echoing fallback; secrets must stay outside chat/logs."""
    try:
        with warnings.catch_warnings():
            warnings.simplefilter('error', getpass.GetPassWarning)
            return getpass.getpass(prompt)
    except getpass.GetPassWarning:
        raise CredentialError('当前终端无法隐藏输入，请在独立终端运行；未读取凭据。') from None


# OWASP Password Storage Cheat Sheet: scrypt when Argon2id is unavailable.
# Fixed parameters cannot be raised by editing an untrusted credential record.
SCRYPT_N = 2 ** 17
SCRYPT_R = 8
SCRYPT_P = 1
MAX_MEMORY = 256 * 1024 * 1024


def _valid_customer(customer_id):
    return isinstance(customer_id, str) and re.fullmatch(r'[a-zA-Z0-9_-]{1,40}', customer_id) is not None


def _password_bytes(password):
    if not isinstance(password, str) or not 1 <= len(password) <= 128:
        return None
    try:
        encoded = password.encode('utf-8')
    except UnicodeError:
        return None
    return encoded if len(encoded) <= 1024 else None


def _derive(password_bytes, salt):
    try:
        return hashlib.scrypt(password_bytes, salt=salt, n=SCRYPT_N,
                              r=SCRYPT_R, p=SCRYPT_P, dklen=32, maxmem=MAX_MEMORY)
    except (ValueError, MemoryError, OverflowError):
        raise CredentialError('本地密码验证无法完成，请检查运行环境。') from None


class AuthStore:
    def __init__(self, directory):
        self.directory = Path(directory)
        self._dummy_salt = secrets.token_bytes(16)

    def create(self, customer_id, password):
        """Create one local classroom account, atomically and without overwrite."""
        if not _valid_customer(customer_id):
            raise CredentialError('模拟账号格式不正确。')
        encoded = _password_bytes(password)
        if encoded is None or len(password) < 15:
            raise CredentialError('课堂密码需要15至128个字符。')
        destination = self.directory / (customer_id + '.json')
        if destination.exists():
            raise CredentialError('此模拟账号已配置密码，未覆盖。')
        salt = secrets.token_bytes(16)
        digest = _derive(encoded, salt)
        record = {'schema_version': 1, 'customer_id': customer_id,
                  'algorithm': 'scrypt', 'salt': salt.hex(), 'digest': digest.hex()}
        temporary = self.directory / ('.' + secrets.token_hex(16) + '.tmp')
        try:
            self.directory.mkdir(parents=True, exist_ok=True)
            with temporary.open('x', encoding='utf-8') as out:
                json.dump(record, out, ensure_ascii=True)
                out.flush()
                os.fsync(out.fileno())
            # A concurrent provision must fail rather than overwrite credentials.
            os.link(temporary, destination)
        except FileExistsError:
            raise CredentialError('此模拟账号已配置密码，未覆盖。') from None
        except OSError:
            raise CredentialError('无法保存本地课堂凭据，未覆盖既有账号。') from None
        finally:
            try:
                temporary.unlink(missing_ok=True)
            except OSError:
                pass

    def _read(self, customer_id):
        path = self.directory / (customer_id + '.json')
        try:
            if path.stat().st_size > 4096:
                raise ValueError('size')
            record = json.loads(path.read_text(encoding='utf-8'))
            fields = {'schema_version', 'customer_id', 'algorithm', 'salt', 'digest'}
            if (not isinstance(record, dict) or set(record) != fields
                    or type(record['schema_version']) is not int or record['schema_version'] != 1
                    or record['customer_id'] != customer_id or record['algorithm'] != 'scrypt'
                    or not isinstance(record['salt'], str) or not re.fullmatch(r'[0-9a-f]{32}', record['salt'])
                    or not isinstance(record['digest'], str) or not re.fullmatch(r'[0-9a-f]{64}', record['digest'])):
                raise ValueError('schema')
            return bytes.fromhex(record['salt']), bytes.fromhex(record['digest'])
        except FileNotFoundError:
            return None
        except (OSError, ValueError, UnicodeError, RecursionError):
            raise CredentialError('本地课堂凭据不可用，请检查配置。') from None

    def verify(self, customer_id, password):
        """Return verified identity or None; invalid/unknown accounts never log in."""
        encoded = _password_bytes(password)
        if not _valid_customer(customer_id) or encoded is None:
            return None
        record = self._read(customer_id)
        # Also derive for unknown accounts; do not turn account existence into a
        # cheaper login path. Both unknown-account and wrong-password yield None.
        salt, expected = record if record is not None else (self._dummy_salt, bytes(32))
        candidate = _derive(encoded, salt)
        matches = hmac.compare_digest(candidate, expected)
        return customer_id if record is not None and matches else None

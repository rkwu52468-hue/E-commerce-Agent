"""Local classroom credential tests; only one case performs real production scrypt."""
from contextlib import redirect_stdout
from copy import deepcopy
import getpass
import hashlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import warnings

from customer_service.auth import AuthStore, CredentialError, read_secret

ROOT = Path(__file__).resolve().parents[1]
PASSWORD = "offline-classroom-long-passphrase"
BASE_RECORD = {"schema_version": 1, "customer_id": "alice", "algorithm": "scrypt",
               "salt": "01" * 16, "digest": "ff" * 32}


class AuthStoreTests(unittest.TestCase):
    def setUp(self):
        (ROOT / "runtime").mkdir(exist_ok=True)
        self.temporary = tempfile.TemporaryDirectory(prefix="auth_store_acceptance_", dir=ROOT / "runtime")
        self.addCleanup(self.temporary.cleanup)
        self.directory = Path(self.temporary.name) / "credentials"
        self.directory.mkdir()
        self.store = AuthStore(self.directory)

    def test_real_scrypt_create_verify_wrong_password_and_unknown_account(self):
        real_scrypt = hashlib.scrypt
        with patch("customer_service.auth.hashlib.scrypt", wraps=real_scrypt) as derive:
            self.store.create("alice", PASSWORD)
            self.assertEqual("alice", self.store.verify("alice", PASSWORD))
            self.assertIsNone(self.store.verify("alice", "offline-classroom-wrong-passphrase"))
            self.assertIsNone(self.store.verify("unknown", PASSWORD))
        self.assertEqual(4, derive.call_count)
        for call in derive.call_args_list:
            self.assertEqual(2 ** 17, call.kwargs["n"])
            self.assertEqual(8, call.kwargs["r"])
            self.assertEqual(1, call.kwargs["p"])
            self.assertEqual(32, call.kwargs["dklen"])
            self.assertEqual(256 * 1024 * 1024, call.kwargs["maxmem"])
            self.assertEqual(16, len(call.kwargs["salt"]))
        raw = (self.directory / "alice.json").read_text(encoding="utf-8")
        self.assertNotIn(PASSWORD, raw)
        record = json.loads(raw)
        self.assertEqual("scrypt", record["algorithm"])
        self.assertEqual(32, len(record["salt"]))
        self.assertEqual(64, len(record["digest"]))

    def test_invalid_customer_or_new_password_fails_before_derivation(self):
        invalid = [(name, PASSWORD) for name in ("", "../bob", "alice/bob", "alice.json", "a" * 41, None)]
        invalid += [("alice", value) for value in ("", "x" * 14, "x" * 129, None, b"x" * 20, "\ud800" * 15)]
        with patch("customer_service.auth.hashlib.scrypt", side_effect=AssertionError("invalid input reached KDF")) as derive:
            for customer, password in invalid:
                with self.subTest(customer=customer, password_type=type(password).__name__), self.assertRaises(CredentialError):
                    self.store.create(customer, password)
            derive.assert_not_called()
        self.assertEqual([], list(self.directory.iterdir()))

    def test_existing_account_file_is_never_overwritten_even_if_malformed(self):
        path = self.directory / "alice.json"
        existing = b"PRIVATE_EXISTING_CREDENTIAL_FILE"
        path.write_bytes(existing)
        with patch("customer_service.auth.hashlib.scrypt", side_effect=AssertionError("duplicate account reached KDF")) as derive:
            with self.assertRaises(CredentialError):
                self.store.create("alice", PASSWORD)
            derive.assert_not_called()
        self.assertEqual(existing, path.read_bytes())

    def test_malformed_or_untrusted_record_cannot_authenticate_or_change_kdf_cost(self):
        variants = [None, [], {**BASE_RECORD, "schema_version": True}, {**BASE_RECORD, "customer_id": "bob"},
                    {**BASE_RECORD, "algorithm": "plaintext"}, {**BASE_RECORD, "salt": "00" * 15},
                    {**BASE_RECORD, "digest": "gg" * 32}, {**BASE_RECORD, "n": 2 ** 35}]
        contents = [json.dumps(item).encode("utf-8") for item in variants]
        contents += [b"PRIVATE_INVALID_JSON_CREDENTIAL", b"x" * 5000]
        path = self.directory / "alice.json"
        with patch("customer_service.auth.hashlib.scrypt", side_effect=AssertionError("bad record reached KDF")) as derive:
            for index, content in enumerate(contents):
                path.write_bytes(content)
                with self.subTest(case=index), self.assertRaises(CredentialError) as caught:
                    self.store.verify("alice", PASSWORD)
                self.assertNotIn(PASSWORD, str(caught.exception))
                self.assertNotIn(str(path), str(caught.exception))
                self.assertNotIn("PRIVATE_INVALID_JSON_CREDENTIAL", str(caught.exception))
            derive.assert_not_called()

    def test_unknown_account_dummy_hash_cannot_become_a_login_even_if_zero_digest_matches(self):
        with patch("customer_service.auth.hashlib.scrypt", return_value=bytes(32)) as derive:
            self.assertIsNone(self.store.verify("nobody", PASSWORD))
        derive.assert_called_once()
        self.assertEqual(2 ** 17, derive.call_args.kwargs["n"])
        self.assertFalse((self.directory / "nobody.json").exists())

    def test_kdf_failure_does_not_expose_exception_body_or_publish_credentials(self):
        marker = "PRIVATE_KDF_FAILURE_DETAIL"
        with patch("customer_service.auth.hashlib.scrypt", side_effect=ValueError(marker)):
            with self.assertRaises(CredentialError) as caught:
                self.store.create("alice", PASSWORD)
        self.assertNotIn(marker, str(caught.exception))
        self.assertNotIn(PASSWORD, str(caught.exception))
        self.assertFalse((self.directory / "alice.json").exists())

    def test_concurrent_publication_does_not_overwrite_the_other_account_file(self):
        path = self.directory / "alice.json"
        concurrent_record = b"ANOTHER_PROVISIONER_ALREADY_CREATED_THIS_RECORD"
        def concurrent_publish(source, destination):
            Path(destination).write_bytes(concurrent_record)
            raise FileExistsError("simulated concurrent provision")
        with patch("customer_service.auth.hashlib.scrypt", return_value=b"d" * 32), \
             patch("customer_service.auth.os.link", side_effect=concurrent_publish):
            with self.assertRaises(CredentialError):
                self.store.create("alice", PASSWORD)
        self.assertEqual(concurrent_record, path.read_bytes())

    def test_secret_reader_returns_hidden_input_without_echoing_or_using_input(self):
        captured = io.StringIO()
        with patch("customer_service.auth.getpass.getpass", return_value=PASSWORD) as hidden, \
             patch("builtins.input", side_effect=AssertionError("echoing input fallback used")), redirect_stdout(captured):
            self.assertEqual(PASSWORD, read_secret("classroom password: "))
        hidden.assert_called_once()
        self.assertNotIn(PASSWORD, captured.getvalue())

    def test_secret_reader_fails_closed_on_getpass_warning_before_echoing_fallback(self):
        continued = []
        def unavailable(prompt):
            warnings.warn("Terminal cannot hide input", getpass.GetPassWarning)
            continued.append(True)
            return PASSWORD
        with patch("customer_service.auth.getpass.getpass", side_effect=unavailable), \
             patch("builtins.input", side_effect=AssertionError("echo fallback used")):
            with self.assertRaises(CredentialError) as caught:
                read_secret("classroom password: ")
        self.assertEqual([], continued)
        self.assertNotIn(PASSWORD, str(caught.exception))
        self.assertNotIn("Terminal cannot hide input", str(caught.exception))


if __name__ == "__main__":
    unittest.main(verbosity=2)

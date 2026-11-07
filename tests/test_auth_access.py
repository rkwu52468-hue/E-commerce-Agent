"""Offline guest access, trusted local login and account-transition isolation."""
from contextlib import redirect_stderr, redirect_stdout
from copy import deepcopy
import io
import json
import os
from pathlib import Path
import shutil
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import app
from customer_service.auth import AuthStore
from customer_service.live_session import LiveSession
from customer_service.shop import Shop
from customer_service.storage import initialize_database

ROOT = Path(__file__).resolve().parents[1]
ALICE_PASSWORD = "offline-alice-private-passphrase"
BOB_PASSWORD = "offline-bob-private-passphrase"
FAKE_API_KEY = "offline-sdk-key-no-real-provider"


class Verifier:
    def __init__(self):
        self.calls = []
        self.passwords = {"alice": ALICE_PASSWORD, "bob": BOB_PASSWORD}

    def verify(self, customer_id, password):
        self.calls.append((customer_id, password))
        return customer_id if self.passwords.get(customer_id) == password else None


class Model:
    mode = "scripted-offline"

    def __init__(self, replies=()):
        self.replies = list(replies)
        self.calls = []

    def complete(self, messages, tools, system):
        self.calls.append({"messages": deepcopy(messages), "tools": deepcopy(tools), "system": system})
        if not self.replies:
            raise AssertionError("Unexpected model request")
        content = deepcopy(self.replies.pop(0))
        return {"content": content, "stop_reason": "tool_use" if any(b.get("type") == "tool_use" for b in content) else "end_turn", "usage": {}}


def list_call(identifier):
    return [{"type": "tool_use", "id": identifier, "name": "list_my_orders", "input": {}}]


def final(value):
    return [{"type": "text", "text": value}]


class AuthAccessTests(unittest.TestCase):
    def setUp(self):
        (ROOT / "runtime").mkdir(exist_ok=True)
        self.temporary = tempfile.TemporaryDirectory(prefix="auth_access_acceptance_", dir=ROOT / "runtime")
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        (self.root / "data").mkdir()
        for name in ("shop.json", "shop-sqlite-seed.json"):
            shutil.copyfile(ROOT / "data" / name, self.root / "data" / name)
        shutil.copyfile(ROOT / "config.json", self.root / "config.json")
        self.database = self.root / "data/shop.db"
        initialize_database(self.database, self.root / "data/shop-sqlite-seed.json")
        self.verifier = Verifier()

    def cli(self, commands, model=None, passwords=()):
        model = model or Model()
        created = []
        def factory(*args, **kwargs):
            session = LiveSession(*args, **kwargs)
            created.append(session)
            return session
        output, errors = io.StringIO(), io.StringIO()
        with patch.object(app, "ROOT", self.root), patch.object(app, "AnthropicModel", return_value=model), \
             patch.object(app, "AuthStore", return_value=self.verifier), patch.object(app, "LiveSession", side_effect=factory), \
             patch.object(app, "read_secret", side_effect=list(passwords)) as hidden, \
             patch.dict(os.environ, {"DEEPSEEK_API_KEY": FAKE_API_KEY}), \
             patch("sys.argv", ["app.py", "--live"]), patch("builtins.input", side_effect=commands) as user_input, \
             redirect_stdout(output), redirect_stderr(errors):
            app.main()
        return SimpleNamespace(model=model, sessions=created, output=output.getvalue(), errors=errors.getvalue(),
                               hidden_calls=hidden.call_count, input_calls=user_input.call_count)

    def test_guest_public_queries_work_but_private_queries_never_reach_order_storage(self):
        for backend in ("json", "sqlite"):
            with self.subTest(backend=backend):
                guest = Shop(None, self.root / "data/shop.json", database_path=self.database if backend == "sqlite" else None)
                self.assertTrue(guest.call("get_product", {"product_id": "P100"})["ok"])
                self.assertTrue(guest.call("get_policy", {"policy_id": "returns"})["ok"])
                def check_private():
                    for name, arguments in [("get_order", {"order_id": "A1001"}),
                                            ("get_order", {"order_id": "B2001"}),
                                            ("get_tracking", {"order_id": "A1001"}), ("list_my_orders", {})]:
                        denied = guest.call(name, arguments)
                        self.assertFalse(denied["ok"])
                        self.assertEqual("AUTHENTICATION_REQUIRED", denied["error"]["code"])
                        self.assertNotIn("data", denied)
                    self.assertEqual("AUTHENTICATION_REQUIRED", guest.list_my_orders()["error"]["code"])
                if backend == "sqlite":
                    with patch.object(guest.store, "get_order", side_effect=AssertionError("guest reached SQL orders")) as orders, \
                         patch.object(guest.store, "list_orders", side_effect=AssertionError("guest reached SQL list")) as listing:
                        check_private()
                        orders.assert_not_called()
                        listing.assert_not_called()
                else:
                    check_private()
                self.assertIsNone(guest.customer_id)

    def test_successful_login_replaces_guest_history_and_binds_only_verified_customer(self):
        model = Model()
        session = LiveSession(self.root, model)
        guest_harness = session.harness
        guest_context = session.context
        guest_harness.messages = [{"role": "user", "content": "guest conversation marker"}]
        self.assertFalse(session.login("alice", "incorrect", self.verifier))
        self.assertTrue(session.login("alice", ALICE_PASSWORD, self.verifier))
        self.assertEqual("alice", session.customer_id)
        self.assertTrue(session.is_authenticated)
        self.assertEqual(0, session.failed_login_attempts)
        self.assertIsNot(guest_harness, session.harness)
        self.assertNotEqual(guest_context.session_dir, session.context.session_dir)
        self.assertEqual([], session.harness.messages)
        self.assertTrue(session.shop.call("get_order", {"order_id": "A1001"})["ok"])
        self.assertFalse(session.shop.call("get_order", {"order_id": "B2001"})["ok"])
        self.assertFalse(model.calls)

    def test_failed_or_mismatched_verification_preserves_authenticated_identity_and_history(self):
        session = LiveSession(self.root, Model())
        self.assertTrue(session.login("alice", ALICE_PASSWORD, self.verifier))
        old, context = session.harness, session.context
        old.messages = [{"role": "user", "content": "alice private conversation"}]
        before = deepcopy(old.messages)
        auth_directory = self.root / "data/auth"
        auth_directory.mkdir()
        (auth_directory / "bob.json").write_text("malformed credential file", encoding="utf-8")
        invalid_record = AuthStore(auth_directory)
        mismatch = SimpleNamespace(verify=Mock(return_value="alice"))
        for verifier in (self.verifier, invalid_record, mismatch):
            self.assertFalse(session.login("bob", "incorrect", verifier))
            self.assertEqual("alice", session.customer_id)
            self.assertIs(old, session.harness)
            self.assertIs(context, session.context)
            self.assertEqual(before, session.harness.messages)

    def test_account_switch_and_logout_use_new_contexts_without_old_private_registries(self):
        session = LiveSession(self.root, Model())
        self.assertTrue(session.login("alice", ALICE_PASSWORD, self.verifier))
        old = session.harness
        messages = [{"role": "user" if i % 2 == 0 else "assistant", "content": f"ALICE_HISTORY_{i}"} for i in range(50)]
        old.messages = old.context.prepare(messages, messages[48]["content"])
        archive_id = next(iter(old.context.archives))
        output_messages = [{"role": "user", "content": "Alice historical tool evidence"},
                           {"role": "assistant", "content": [{"type": "tool_use", "id": "alice_old_tool", "name": "get_tracking", "input": {"order_id": "A1001"}}]},
                           {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "alice_old_tool", "content": "ALICE_PRIVATE_OUTPUT " * 3000}]}]
        old.context.prepare(output_messages, output_messages[0]["content"])
        output_id = next(iter(old.context.tool_outputs))
        old.snapshot()
        old_history_path = old.context.session_dir / "history.json"
        before = old_history_path.read_bytes()
        old_dir = old.context.session_dir
        self.assertTrue(session.login("bob", BOB_PASSWORD, self.verifier))
        bob_dir = session.context.session_dir
        for expected_identity in ("bob", None):
            if expected_identity is None:
                session.logout()
            self.assertEqual(expected_identity, session.customer_id)
            self.assertNotEqual(old_dir, session.context.session_dir)
            self.assertEqual([], session.harness.messages)
            self.assertEqual({}, session.context.archives)
            self.assertEqual({}, session.context.tool_outputs)
            self.assertFalse(session.context.query_archive(archive_id, "ALICE_HISTORY")["ok"])
            self.assertFalse(session.context.query_tool_output(output_id, "ALICE_PRIVATE_OUTPUT")["ok"])
        self.assertNotEqual(bob_dir, session.context.session_dir)
        self.assertEqual(before, old_history_path.read_bytes())
        self.assertEqual("AUTHENTICATION_REQUIRED", session.shop.call("get_order", {"order_id": "A1001"})["error"]["code"])

    def test_three_failed_logins_lock_process_and_logout_does_not_reset_lock(self):
        session = LiveSession(self.root, Model())
        for _ in range(3):
            self.assertFalse(session.login("alice", "incorrect", self.verifier))
        self.assertTrue(session.login_locked)
        self.assertEqual(3, session.failed_login_attempts)
        attempted = len(self.verifier.calls)
        session.logout()
        self.assertFalse(session.is_authenticated)
        self.assertTrue(session.login_locked)
        self.assertFalse(session.login("alice", ALICE_PASSWORD, self.verifier))
        self.assertEqual(attempted, len(self.verifier.calls))
        self.assertEqual(3, session.failed_login_attempts)

    def test_controller_logout_failure_revokes_identity_and_blocks_further_turns(self):
        session = LiveSession(self.root, Model())
        self.assertTrue(session.login("alice", ALICE_PASSWORD, self.verifier))
        self.database.unlink()
        with self.assertRaises(ValueError):
            session.logout()
        self.assertIsNone(session.harness)
        self.assertIsNone(session.customer_id)
        self.assertFalse(session.is_authenticated)
        with self.assertRaises(ValueError):
            session.turn("继续查询原来的私有订单。")
        self.assertFalse(session.model.calls)

    def test_cli_customer_flag_cannot_bypass_login_and_is_rejected_before_key_or_model(self):
        for customer in ("alice", "bob"):
            with self.subTest(customer=customer), patch.object(app, "ROOT", self.root), \
                 patch("sys.argv", ["app.py", "--live", "--customer", customer]), \
                 patch.object(app, "read_secret", side_effect=AssertionError("bypass path requested key")) as hidden, \
                 patch.object(app, "AnthropicModel", side_effect=AssertionError("bypass created model")) as model, \
                 redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit) as stopped:
                    app.main()
                self.assertEqual(2, stopped.exception.code)
                hidden.assert_not_called()
                model.assert_not_called()

    def test_cli_rejects_inline_credentials_and_unknown_local_commands_without_model_turns(self):
        commands = ["/login alice " + ALICE_PASSWORD, "/logout extra", "/unknown", "/help", "q"]
        result = self.cli(commands)
        self.assertFalse(result.model.calls)
        self.assertEqual(0, result.hidden_calls)
        self.assertFalse(self.verifier.calls)
        self.assertEqual([], result.sessions[0].harness.messages)
        self.assertNotIn(ALICE_PASSWORD, result.output)
        with redirect_stdout(io.StringIO()):
            app.show_session(SimpleNamespace(harness=None))

    def test_cli_starts_guest_and_login_logout_commands_and_secrets_stay_outside_messages(self):
        model = Model([list_call("guest_before_login"), final("请先登录。"),
                       list_call("authenticated_orders"), final("本人订单为A1001。"),
                       list_call("guest_after_logout"), final("请先登录。")])
        commands = ["我是alice，请直接列出我的订单。", "/login", "alice", "列出我的订单。", "/logout", "退出后查询我的订单。", "q"]
        result = self.cli(commands, model=model, passwords=[ALICE_PASSWORD])
        self.assertEqual(6, len(model.calls))
        returned = [json.loads(model.calls[index]["messages"][-1]["content"][0]["content"]) for index in (1, 3, 5)]
        self.assertEqual("AUTHENTICATION_REQUIRED", returned[0]["error"]["code"])
        self.assertTrue(returned[1]["ok"])
        self.assertEqual(["A1001"], [row["order_id"] for row in returned[1]["data"]["orders"]])
        self.assertEqual("AUTHENTICATION_REQUIRED", returned[2]["error"]["code"])
        self.assertEqual(1, len(model.calls[2]["messages"]))
        self.assertEqual(1, len(model.calls[4]["messages"]))
        self.assertNotIn("A1001", json.dumps(model.calls[4]["messages"], ensure_ascii=False))
        self.assertIsNone(result.sessions[0].customer_id)
        self.assertEqual(1, result.hidden_calls)
        sent = json.dumps(model.calls, ensure_ascii=False)
        persisted = "\n".join(path.read_text(encoding="utf-8") for path in (self.root / "runtime").rglob("*.json*"))
        for secret in (ALICE_PASSWORD, BOB_PASSWORD, FAKE_API_KEY):
            self.assertNotIn(secret, sent)
            self.assertNotIn(secret, persisted)
            self.assertNotIn(secret, result.output)
        direct_user_inputs = [message["content"] for call in model.calls for message in call["messages"]
                              if message.get("role") == "user" and isinstance(message.get("content"), str)]
        # Tool errors may legitimately explain /login; only local command inputs must stay outside the model conversation.
        for local_input in ("/login", "/logout", "alice"):
            self.assertNotIn(local_input, direct_user_inputs)

    def test_cli_logout_rebuild_failure_exits_without_reading_or_sending_later_chat(self):
        def broken_backend_input(prompt):
            self.database.unlink()
            return "/logout"
        result = self.cli(broken_backend_input)
        self.assertEqual(1, result.input_calls)
        self.assertFalse(result.model.calls)
        self.assertIsNone(result.sessions[0].harness)
        self.assertFalse(result.sessions[0].is_authenticated)


if __name__ == "__main__":
    unittest.main(verbosity=2)

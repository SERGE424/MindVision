#!/usr/bin/env python
import json
import tempfile
import unittest
from pathlib import Path

from AssistantIA_Complet import AssistantIA


class TestConversationsAndUiSettings(unittest.TestCase):
    def setUp(self):
        self.tmp_dir_obj = tempfile.TemporaryDirectory()
        self.tmp_dir = Path(self.tmp_dir_obj.name)

        self.app = AssistantIA.__new__(AssistantIA)
        self.app.ui_settings_path = self.tmp_dir / "ui_settings_32_cartes.json"
        self.app.conversations_dir = self.tmp_dir / "conversations"
        self.app.conversations_dir.mkdir(parents=True, exist_ok=True)
        self.app.conversations = {}
        self.app.last_image_dir_chat = "DEFAULT"

    def tearDown(self):
        self.tmp_dir_obj.cleanup()

    def _write_json(self, file_path, payload):
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)

    def test_ui_settings_save_and_reload_last_image_dir(self):
        target_dir = self.tmp_dir / "images"
        target_dir.mkdir(exist_ok=True)

        # Simule un settings existant pour vérifier qu'on conserve les autres clés.
        self._write_json(self.app.ui_settings_path, {"sash_y": 497})

        self.app.last_image_dir_chat = str(target_dir)
        self.app._sauvegarder_ui_settings()

        with open(self.app.ui_settings_path, "r", encoding="utf-8") as f:
            saved = json.load(f)

        self.assertEqual(saved.get("sash_y"), 497)
        self.assertEqual(saved.get("last_image_dir_chat"), str(target_dir))

        # Réinitialise et recharge pour valider la persistance.
        self.app.last_image_dir_chat = "DEFAULT"
        self.app._charger_ui_settings()
        self.assertEqual(self.app.last_image_dir_chat, str(target_dir))

    def test_ui_settings_missing_or_invalid_file_does_not_crash(self):
        # Fichier manquant: doit rester sur la valeur par défaut.
        self.app._charger_ui_settings()
        self.assertEqual(self.app.last_image_dir_chat, "DEFAULT")

        # Fichier invalide: doit aussi rester sur la valeur par défaut.
        with open(self.app.ui_settings_path, "w", encoding="utf-8") as f:
            f.write("{invalid_json")

        self.app.last_image_dir_chat = "DEFAULT"
        self.app._charger_ui_settings()
        self.assertEqual(self.app.last_image_dir_chat, "DEFAULT")

    def test_conversations_loader_filters_non_chatbot_and_normalizes(self):
        self._write_json(
            self.app.conversations_dir / "2026-04-23_18-01-02.json",
            {
                "nom": "Conversation test",
                "date_creation": "2026-04-23T18:01:02",
                "messages": [
                    {"sender": "Vous", "texte": "Bonjour", "timestamp": "2026-04-23T18:01:03"}
                ],
            },
        )

        # Ancien export chatbot sans nom/date mais avec sender/texte.
        self._write_json(
            self.app.conversations_dir / "2026-04-23_legacy_sender.json",
            {
                "messages": [
                    {"sender": "Vous", "texte": "Texte legacy", "timestamp": "2026-04-23T18:02:00"}
                ]
            },
        )

        # Fichier non chatbot de type chat_cartes.
        self._write_json(
            self.app.conversations_dir / "chat_cartes_2026-04-23_09-25-45.json",
            {
                "session_started": "2026-04-23T09:25:45",
                "messages": [{"role": "Assistant", "message": "Hello"}],
            },
        )

        # Historique tableau: non chatbot.
        with open(
            self.app.conversations_dir / "citations_history_32_cartes.json",
            "w",
            encoding="utf-8",
        ) as f:
            json.dump([{"citation": "x"}], f, ensure_ascii=False)

        self.app.charger_conversations()

        self.assertIn("2026-04-23_18-01-02", self.app.conversations)
        self.assertIn("2026-04-23_legacy_sender", self.app.conversations)
        self.assertNotIn("chat_cartes_2026-04-23_09-25-45", self.app.conversations)
        self.assertNotIn("citations_history_32_cartes", self.app.conversations)

        legacy = self.app.conversations["2026-04-23_legacy_sender"]
        self.assertIn("nom", legacy)
        self.assertIn("date_creation", legacy)
        self.assertIsInstance(legacy["messages"], list)
        self.assertEqual(legacy["messages"][0]["sender"], "Vous")
        self.assertEqual(legacy["messages"][0]["texte"], "Texte legacy")

    def test_chat_cartes_file_is_explicitly_not_chatbot(self):
        result = self.app._est_fichier_conversation_chatbot(
            Path("chat_cartes_2026-04-23_10-00-00.json"),
            {"messages": [{"sender": "Vous", "texte": "Test"}]},
        )
        self.assertFalse(result)

    def test_traiter_commande_chat_accepte_alias_ronommer(self):
        captured = {"title": None}

        def fake_rename(nouveau_titre=None):
            captured["title"] = nouveau_titre

        self.app._on_renommer_conversation = fake_rename

        handled = self.app._traiter_commande_chat("ronommer Projet Atlas")

        self.assertTrue(handled)
        self.assertEqual(captured["title"], "Projet Atlas")

    def test_traiter_commande_chat_ignore_si_image_jointe(self):
        captured = {"called": False}

        def fake_rename(nouveau_titre=None):
            captured["called"] = True

        self.app._on_renommer_conversation = fake_rename

        handled = self.app._traiter_commande_chat("renommer Titre", image_path="image.png")

        self.assertFalse(handled)
        self.assertFalse(captured["called"])


if __name__ == "__main__":
    unittest.main(verbosity=2)

from __future__ import annotations

import json
import unittest

from research_factory.death_census_flatten import FlattenError, flatten_body


class DeathCensusFlattenTest(unittest.TestCase):
    def test_root_level_blocks_are_separated_without_splitting_inline_text(self):
        body = json.dumps(
            {
                "node": "root",
                "child": [
                    {
                        "node": "element",
                        "tag": "a",
                        "child": [{"node": "text", "text": "First sentence."}],
                    },
                    {
                        "node": "element",
                        "tag": "p",
                        "child": [
                            {"node": "text", "text": "Second "},
                            {
                                "node": "element",
                                "tag": "strong",
                                "child": [{"node": "text", "text": "sentence."}],
                            },
                        ],
                    },
                ],
            }
        )
        self.assertEqual(flatten_body(body), "First sentence. Second sentence.")

    def test_nested_rich_text_nodes_preserve_document_order(self):
        body = json.dumps(
            {
                "node": "root",
                "child": [
                    {
                        "node": "element",
                        "tag": "p",
                        "child": [
                            {"node": "text", "text": "Binance will "},
                            {
                                "node": "element",
                                "tag": "strong",
                                "child": [{"node": "text", "text": "rename"}],
                            },
                            {"node": "text", "text": " OLD as NEW."},
                        ],
                    },
                    {
                        "node": "element",
                        "tag": "p",
                        "child": [{"node": "text", "text": "Second paragraph."}],
                    },
                ],
            }
        )
        self.assertEqual(
            flatten_body(body), "Binance will rename OLD as NEW. Second paragraph."
        )

    def test_html_and_entities_are_flattened_without_content_change(self):
        self.assertEqual(flatten_body("<p>A&nbsp;B</p><p>C &amp; D</p>"), "A B C & D")

    def test_empty_flattened_body_fails_closed(self):
        with self.assertRaisesRegex(FlattenError, "empty"):
            flatten_body('{"node":"root","child":[]}')


if __name__ == "__main__":
    unittest.main()

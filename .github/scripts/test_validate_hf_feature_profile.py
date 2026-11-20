import json
import unittest

from build_training_manifest import _self_test_profile
from validate_hf_feature_profile import (
    ProfileCheckError,
    _source_url,
    is_candidate_path,
    list_candidate_paths,
    validate_profile_bytes,
)


class HFProfileValidationTests(unittest.TestCase):
    def test_candidate_path_filter(self):
        self.assertTrue(is_candidate_path("feature_context_profile.json"))
        self.assertTrue(is_candidate_path("profiles/contexts.yaml"))
        self.assertFalse(is_candidate_path("README.md"))
        self.assertFalse(is_candidate_path("holdout.sqlite"))

    def test_profile_json_must_match_manifest_schema(self):
        profile = _self_test_profile()
        parsed = validate_profile_bytes(json.dumps(profile).encode())
        self.assertEqual(parsed["source_revision"], "a" * 40)
        self.assertEqual(parsed["source_path"], "native/tests/support/feature_contexts.rs")
        with self.assertRaises(ProfileCheckError):
            validate_profile_bytes(b'{"schema":"wrong"}')

    def test_source_url_uses_pinned_revision_and_relative_path(self):
        url = _source_url("a" * 40, "native/tests/support/feature_contexts.rs")
        self.assertIn("a" * 40, url)
        self.assertTrue(url.endswith("native/tests/support/feature_contexts.rs"))

    def test_profile_tree_search_handles_pagination(self):
        second = "https://huggingface.co/api/datasets/example/tree/main?cursor=2"
        calls = []

        def fetch(url):
            calls.append(url)
            if "cursor=" in url:
                return ["profiles/feature_contexts.json"], None
            return ["README.md", "holdout.sqlite"], second

        paths = list_candidate_paths("a" * 40, fetch_page=fetch)

        self.assertEqual(paths, ["profiles/feature_contexts.json"])
        self.assertEqual(len(calls), 2)

    def test_invalid_revision_and_repeated_pagination_fail(self):
        with self.assertRaises(ProfileCheckError):
            list_candidate_paths("main", fetch_page=lambda _: ([], None))
        repeated = "https://huggingface.co/api/datasets/example/tree/main"
        with self.assertRaises(ProfileCheckError):
            list_candidate_paths("a" * 40, fetch_page=lambda _: ([], repeated))


if __name__ == "__main__":
    unittest.main()

import json
import unittest

from build_training_manifest import _self_test_profile
from validate_hf_feature_profile import (
    ProfileCheckError,
    _source_exists,
    _source_url,
    inspect_profile,
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

    def test_inspect_profile_skips_unrelated_json_and_finds_one_pinned_profile(self):
        profile = _self_test_profile()
        paths = [
            "profiles/authoritative_context_profile.json",
            "examples/context_example.json",
        ]

        def fetch_file(url, _limit, _allowed_host):
            if url.endswith("authoritative_context_profile.json"):
                return json.dumps(profile).encode()
            return b'{"schema":"unrelated"}'

        status = inspect_profile(
            fetch_page=lambda _url: (paths, None),
            fetch_file=fetch_file,
            source_exists=lambda revision, path: (
                revision == "a" * 40
                and path == "native/tests/support/feature_contexts.rs"
            ),
            revision_getter=lambda: "b" * 40,
        )

        self.assertEqual(status, "schema_source_pinned")

    def test_inspect_profile_accepts_equivalent_pinned_profiles(self):
        first = _self_test_profile()
        second = _self_test_profile()
        second["profile_id"] = "alternate-self-test"
        profiles = {
            "profiles/first_context_profile.json": json.dumps(first).encode(),
            "profiles/second_feature_profile.json": json.dumps(second).encode(),
        }
        paths = list(profiles)

        def fetch_file(url, _limit, _allowed_host):
            for path, payload in profiles.items():
                if url.endswith(path):
                    return payload
            self.fail("unexpected candidate URL")

        status = inspect_profile(
            fetch_page=lambda _url: (paths, None),
            fetch_file=fetch_file,
            source_exists=lambda _revision, _path: True,
            revision_getter=lambda: "b" * 40,
        )

        self.assertEqual(status, "schema_source_pinned")

    def test_inspect_profile_rejects_distinct_pinned_profiles(self):
        first = _self_test_profile()
        second = _self_test_profile()
        second["profile_id"] = "alternate-self-test"
        second["feature_contexts"]["max_parity_patterns"] += 1
        profiles = {
            "profiles/first_context_profile.json": json.dumps(first).encode(),
            "profiles/second_feature_profile.json": json.dumps(second).encode(),
        }
        paths = list(profiles)

        def fetch_file(url, _limit, _allowed_host):
            for path, payload in profiles.items():
                if url.endswith(path):
                    return payload
            self.fail("unexpected candidate URL")

        status = inspect_profile(
            fetch_page=lambda _url: (paths, None),
            fetch_file=fetch_file,
            source_exists=lambda _revision, _path: True,
            revision_getter=lambda: "b" * 40,
        )

        self.assertEqual(status, "candidate_ambiguous")

    def test_missing_pinned_source_is_unverified_not_unavailable(self):
        def missing_source(*_args):
            raise ProfileCheckError("source file was not found")

        self.assertFalse(
            _source_exists(
                "a" * 40,
                "native/tests/support/feature_contexts.rs",
                fetch_file=missing_source,
            )
        )


if __name__ == "__main__":
    unittest.main()

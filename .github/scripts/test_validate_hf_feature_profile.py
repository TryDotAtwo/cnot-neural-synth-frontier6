import json
import unittest

from build_training_manifest import _self_test_profile
from validate_hf_feature_profile import (
    ProfileCheckError,
    _select_pinned_profile,
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
        self.assertTrue(is_candidate_path("profiles/contexts.json"))
        self.assertTrue(is_candidate_path("profiles/train.json"))
        self.assertTrue(is_candidate_path("feature_contexts/profile.json"))
        self.assertTrue(is_candidate_path("feature_contexts/train.json"))
        self.assertFalse(is_candidate_path("contexts/records/example.json"))
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

    def test_inspect_profile_limits_candidate_json_count(self):
        paths = [f"profiles/context_profile_{index}.json" for index in range(33)]

        status = inspect_profile(
            fetch_page=lambda _url: (paths, None),
            fetch_file=lambda *_args: self.fail("candidate limit must stop before downloads"),
            revision_getter=lambda: "b" * 40,
        )

        self.assertEqual(status, "candidate_limit_exceeded")

    def test_invalid_revision_and_repeated_pagination_fail(self):
        with self.assertRaises(ProfileCheckError):
            list_candidate_paths("main", fetch_page=lambda _: ([], None))
        repeated = "https://huggingface.co/api/datasets/example/tree/main"
        with self.assertRaises(ProfileCheckError):
            list_candidate_paths("a" * 40, fetch_page=lambda _: ([], repeated))

    def test_inspect_profile_tries_explicit_profile_names_before_context_folder(self):
        profile = _self_test_profile()
        payloads = {
            "metadata/unrelated_profile.json": b'{"schema":"wrong"}',
            "feature_contexts/context.json": json.dumps(profile).encode(),
        }

        def fetch_file(url, _limit, _allowed_host):
            for path, payload in payloads.items():
                if url.endswith(path):
                    return payload
            self.fail("unexpected candidate URL")

        status = inspect_profile(
            fetch_page=lambda _url: (list(payloads), None),
            fetch_file=fetch_file,
            source_exists=lambda _revision, _path: True,
            revision_getter=lambda: "b" * 40,
        )

        self.assertEqual(status, "schema_source_pinned")

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

        self.assertEqual(status, "candidate_contexts_incomparable")

    def test_select_profile_prefers_exact_context_extension(self):
        base_profile = _self_test_profile()
        extended_profile = _self_test_profile()
        extended_profile["feature_contexts"]["labeled_cuts"].append(
            {"n": 4, "a": [0]}
        )
        base = validate_profile_bytes(json.dumps(base_profile).encode())
        extended = validate_profile_bytes(json.dumps(extended_profile).encode())

        selected = _select_pinned_profile(
            [
                ("profiles/base_context_profile.json", base),
                ("profiles/extended_context_profile.json", extended),
            ]
        )

        self.assertIsNotNone(selected)
        self.assertEqual(selected[0], "profiles/extended_context_profile.json")


    def test_inspect_profile_reports_distinct_source_pins(self):
        first = _self_test_profile()
        second = _self_test_profile()
        second["source"]["revision"] = "c" * 40
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

        self.assertEqual(status, "candidate_source_ambiguous")


    def test_inspect_profile_reports_safe_failure_stage(self):
        def fail(message):
            raise ProfileCheckError(message)

        self.assertEqual(
            inspect_profile(revision_getter=lambda: fail("Hub revision request failed")),
            "unavailable_revision",
        )
        self.assertEqual(
            inspect_profile(
                revision_getter=lambda: "b" * 40,
                fetch_page=lambda _url: fail("Hub tree request failed"),
            ),
            "unavailable_tree",
        )
        for message, expected in (
            ("download request failed", "unavailable_profile_request_failed"),
            ("source file was not found", "unavailable_profile_not_found"),
            ("download size limit exceeded", "unavailable_profile_size_limit"),
            ("download redirect host is invalid", "unavailable_profile_redirect_rejected"),
        ):
            with self.subTest(message=message):
                self.assertEqual(
                    inspect_profile(
                        revision_getter=lambda: "b" * 40,
                        fetch_page=lambda _url: (
                            ["profiles/feature_context_profile.json"], None
                        ),
                        fetch_file=lambda *_args: fail(message),
                    ),
                    expected,
                )
        self.assertEqual(
            inspect_profile(
                revision_getter=lambda: "b" * 40,
                fetch_page=lambda _url: (
                    ["profiles/feature_context_profile.json"], None
                ),
                fetch_file=lambda *_args: json.dumps(_self_test_profile()).encode(),
                source_exists=lambda *_args: fail("download request failed"),
            ),
            "unavailable_source_request_failed",
        )

    def test_profile_reports_objective_alphabet_mismatch_as_safe_status(self):
        profile = _self_test_profile()
        n = 3
        gates = [
            {"control": control, "target": target}
            for control in range(n)
            for target in range(n)
            if control != target
        ]
        profile["feature_contexts"]["alphabet"] = {
            "dimension": n,
            "id": f"all_to_all_unit_cnot:{n}",
            "gates": gates,
            "costs": [1] * len(gates),
        }
        status = inspect_profile(
            fetch_page=lambda _url: (
                ["profiles/feature_context_profile.json"], None
            ),
            fetch_file=lambda *_args: json.dumps(profile).encode(),
            source_exists=lambda *_args: self.fail("invalid profile must not reach source checks"),
            revision_getter=lambda: "b" * 40,
        )
        self.assertEqual(status, "candidate_objective_alphabet_incompatible")

    def test_inspect_profile_classifies_invalid_candidate_schema(self):
        status = inspect_profile(
            fetch_page=lambda _url: (["profiles/candidate_profile.json"], None),
            fetch_file=lambda *_args: b'{"schema":"wrong"}',
            revision_getter=lambda: "b" * 40,
        )

        self.assertEqual(status, "candidate_schema_unverified")

    def test_inspect_profile_classifies_missing_pinned_source(self):
        profile = _self_test_profile()
        status = inspect_profile(
            fetch_page=lambda _url: (["profiles/candidate_profile.json"], None),
            fetch_file=lambda *_args: json.dumps(profile).encode(),
            source_exists=lambda _revision, _path: False,
            revision_getter=lambda: "b" * 40,
        )

        self.assertEqual(status, "candidate_source_unverified")

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

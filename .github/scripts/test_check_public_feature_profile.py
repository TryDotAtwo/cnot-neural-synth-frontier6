import unittest

from check_public_feature_profile import (
    MetadataQueryError,
    find_profile_candidate,
    is_profile_candidate,
    next_page_url,
)


class ProfilePathTests(unittest.TestCase):
    def test_recognizes_only_small_profile_or_context_config_paths(self):
        self.assertTrue(is_profile_candidate("profiles/feature_contexts.json"))
        self.assertTrue(is_profile_candidate("training/context-profile.yaml"))
        self.assertTrue(is_profile_candidate("profile.toml"))
        self.assertFalse(is_profile_candidate("full_shared_native_holdout_20261001.sqlite"))
        self.assertFalse(is_profile_candidate("README.md"))
        self.assertFalse(is_profile_candidate("context_notes.txt"))

    def test_next_link_accepts_only_hugging_face_hosts(self):
        link = '<https://huggingface.co/api/datasets/x/tree/main?cursor=next>; rel="next"'
        self.assertEqual(
            next_page_url(link, "https://huggingface.co/api/datasets/x/tree/main"),
            "https://huggingface.co/api/datasets/x/tree/main?cursor=next",
        )
        with self.assertRaises(MetadataQueryError):
            next_page_url('<https://example.org/steal>; rel="next"', "https://huggingface.co/tree")

    def test_search_follows_pages_without_exposing_paths(self):
        second = "https://huggingface.co/api/datasets/x/tree/main?cursor=2"
        calls = []

        def fetch(url):
            calls.append(url)
            if "cursor=" in url:
                return ["profiles/feature_contexts.json"], None
            return ["README.md", "holdout.sqlite"], second

        status = find_profile_candidate("a" * 40, fetch_page=fetch)

        self.assertEqual(status, "candidate_found")
        self.assertEqual(len(calls), 2)
        self.assertIn("cursor=2", calls[1])

    def test_not_found_requires_all_pages_to_finish(self):
        second = "https://huggingface.co/api/datasets/x/tree/main?cursor=2"
        calls = []

        def fetch(url):
            calls.append(url)
            if "cursor=" in url:
                return ["holdout.sqlite"], None
            return ["README.md"], second

        self.assertEqual(
            find_profile_candidate("a" * 40, fetch_page=fetch),
            "not_found",
        )
        self.assertEqual(len(calls), 2)

    def test_invalid_revision_and_repeated_pagination_fail_closed(self):
        with self.assertRaises(MetadataQueryError):
            find_profile_candidate("main", fetch_page=lambda _: ([], None))
        url = "https://huggingface.co/api/datasets/x/tree/main"
        with self.assertRaises(MetadataQueryError):
            find_profile_candidate("a" * 40, fetch_page=lambda _: ([], url))


if __name__ == "__main__":
    unittest.main()

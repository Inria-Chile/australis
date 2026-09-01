from polarfunc_repro.download import manifest_from_zenodo_payload


def test_zenodo_payload_becomes_sorted_checksummed_manifest():
    payload = {
        "id": 14181291,
        "metadata": {"license": {"id": "cc-by-4.0"}},
        "files": [
            {
                "key": "z.fa.gz",
                "size": 20,
                "checksum": "md5:" + "a" * 32,
                "links": {"content": "https://zenodo.org/api/records/1/files/z.fa.gz/content"},
            },
            {
                "key": "a.tsv.gz",
                "size": 10,
                "checksum": "md5:" + "b" * 32,
                "links": {"content": "https://zenodo.org/api/records/1/files/a.tsv.gz/content"},
            },
        ],
    }

    manifest = manifest_from_zenodo_payload(payload)

    assert [item.artifact_id for item in manifest.artifacts] == ["a.tsv.gz", "z.fa.gz"]
    assert manifest.artifacts[0].md5 == "b" * 32
    assert manifest.artifacts[0].license == "cc-by-4.0"

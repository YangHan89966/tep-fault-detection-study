"""Run the archived study with its English edition of the predecessor protocol."""

import run_study_frozen as study


PROTOCOL_PATH = "learning/assets/HISTORICAL_PROTOCOL.json"
HISTORICAL_PROTOCOL_SHA256 = "264dfb33d9f77e7159a686997768fcca2b042cde6cbb22187a694f46ddf222af"
PUBLIC_PROTOCOL_SHA256 = "75e63e680f208b15ef3f7ab60e3be814581d305cc4b04acc1b1bd7a04c81f06a"


def verify_public_code():
    """Check every historical binding, with one separately pinned documentation edition."""
    lock_path = study.HERE / "locks/CODE_LOCK.json"
    lock = study.get(lock_path)
    study.need(lock["design_lock"] == study.LOCK, "Wrong numerical release design")
    study.need(lock["files"].get(PROTOCOL_PATH) == HISTORICAL_PROTOCOL_SHA256,
               "Unrecognized predecessor protocol in the original code lock")
    for name, digest in lock["files"].items():
        source = study.HERE / name
        if name == "run_study.py":
            source = study.HERE / "run_study_frozen.py"
        if name == PROTOCOL_PATH:
            digest = PUBLIC_PROTOCOL_SHA256
        study.need(study.sha(source) == digest, "Changed archive member: " + name)
    return study.sha(lock_path)


def __getattr__(name):
    return getattr(study, name)


study.verify_code = verify_public_code

if __name__ == "__main__":
    study.main()

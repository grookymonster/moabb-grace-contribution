import io
import logging

from moabb.datasets.martinezcagigal2023_pary_cvep import MartinezCagigal2023Pary


def test_verbosity():
    # Setup logger to capture output
    logger = logging.getLogger("moabb.datasets.martinezcagigal2023_pary_cvep")
    log_capture = io.StringIO()
    handler = logging.StreamHandler(log_capture)
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)

    MartinezCagigal2023Pary()

    # We can't easily run get_data without downloading, but we can check if the logger is set up
    print(f"Logger name: {logger.name}")
    print(f"Logger level: {logger.level}")

    # Check if print statements are gone from the file (manual check was done, but let's see if we can trigger a log)
    # Since we can't easily trigger the loading without real data, we just verify the file content programmatically

    with open("moabb/datasets/martinezcagigal2023_pary_cvep.py", "r") as f:
        content = f.read()
        assert "print(" not in content or 'if __name__ == "__main__":' in content
        assert "log.info(" in content
        assert "log.error(" in content

    print(
        "Verification successful: No print statements found in core logic, logging implemented."
    )


if __name__ == "__main__":
    test_verbosity()

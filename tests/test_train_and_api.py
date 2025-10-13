import json
import subprocess
import sys
from pathlib import Path

import httpx
import pytest
import uvicorn

ART = Path("artifacts/model.pkl")

def test_training_script_creates_artifacts

from unittest.mock import patch, MagicMock
import json
import pytest
from models import angle_model

ERR_STR = "No car on photo or bad angle, cannot change background"


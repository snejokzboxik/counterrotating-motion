from __future__ import annotations
import json
import sys
import unittest
from pathlib import Path

BASE=Path(__file__).resolve().parents[1]
ROOT=BASE.parents[2]
sys.path.insert(0,str(BASE/"code"))
sys.path.insert(0,str(ROOT/"research"/"v17c"/"v17c_21"/"code"))
import analytical_model as legacy
from m3_continuation import change_field_parameter

class StrictDataAndRFMutationTests(unittest.TestCase):
    def test_amplitude_parameters_scale_entire_rf_family(self):
        d=legacy.load_inputs(); changed=change_field_parameter(d,"A",1.1*d["modes"][0].amplitude)
        old={m.k:m.amplitude for m in d["modes"]}; new={m.k:m.amplitude for m in changed["modes"]}
        for mode in d["modes"]:
            self.assertAlmostEqual(new[mode.k],old[mode.k]*(1.1 if mode.family==0 else 1.0),places=14)

    def test_nu_updates_all_family_two_carriers_only(self):
        d=legacy.load_inputs(); old={m.k:m.frequency for m in d["modes"]}
        changed=change_field_parameter(d,"nu",1.02*old[next(m.k for m in d["modes"] if m.family==1)])
        for mode in changed["modes"]:
            self.assertAlmostEqual(mode.frequency,1.02*old[mode.k] if mode.family==1 else old[mode.k],places=14)

    def test_json_encoder_is_strict(self):
        for bad in (float("nan"),float("inf"),-float("inf")):
            with self.assertRaises(ValueError):
                json.dumps({"diagnostic":bad},allow_nan=False)
        sample={"status":"UNRESOLVED","diagnostic":None}
        self.assertEqual(json.loads(json.dumps(sample,allow_nan=False)),sample)

    def test_all_current_result_json_is_strict(self):
        for path in (BASE/"results").glob("*.json"):
            json.loads(path.read_text(encoding="utf-8"),parse_constant=lambda token: (_ for _ in ()).throw(ValueError(token)))

if __name__=="__main__": unittest.main()

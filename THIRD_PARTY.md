# Third-party code and release review

This repository was organized from the local `RFFI_OSDA_DSCG` research project.
It is an implementation of CWODA, not the official release of UADAL or of the
other upstream projects mentioned below. Existing source-file attribution
comments have been retained.

| Code area | Attribution recorded in the inherited source | Upstream |
| --- | --- | --- |
| Inherited domain-adaptation framework | UADAL | <https://github.com/JoonHo-Jang/UADAL> |
| `models/basenet.py`, `data_loader/mydataset.py`, portions of `utils/utils.py` | OPDA_BP | <https://github.com/ksaito-ut/OPDA_BP> |
| `data_loader/base.py` | SENTRY | <https://github.com/virajprabhu/SENTRY> |
| Portions of `models/function.py` | LabelNoiseCorrection and Separate_to_Adapt | <https://github.com/PaulAlbert31/LabelNoiseCorrection>, <https://github.com/thuml/Separate_to_Adapt> |

The inherited `LICENSE` file is included for review. **Before publishing this
repository, verify the licensing and redistribution terms for each inherited
source file, especially any upstream project without an explicit license.**
The presence of `LICENSE` in this folder does not establish rights over code
originating in another project. Add any required upstream notices or obtain
permission before making the repository public.

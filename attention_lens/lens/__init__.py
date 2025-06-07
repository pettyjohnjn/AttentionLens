"""
To see all the implemented ``Lens``, simply import ``Lens`` and print ``Lens.registry``.

```mermaid
flowchart LR
    Lens
    subgraph Registry
       LensA
    end

    Lens-->LensA
```
"""

import os

current_dir = os.getcwd()
print(current_dir)


from attention_lens.lens.base import Lens
from attention_lens.lens.registry.lensA import LensA
from attention_lens.lens.registry.lensLR import LensLR
from attention_lens.lens.registry.loRA_Lens import LoRA_Lens
from attention_lens.lens.registry.old_lensLR import OldLensLR

__all__ = ["Lens", "LensA", "LensLR", "LoRA_Lens", "OldLensLR"]
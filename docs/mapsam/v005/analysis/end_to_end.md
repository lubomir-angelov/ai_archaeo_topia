## End-to-end: detector candidates through the frozen 512 decoder

Denominator is annotations with geometry on the evaluation sheet, not
segmentation samples and not all 180 annotations.

| fold | sheet | arm | n | IoU>=0.5 | IoU>=0.75 | mean IoU | missed | false masks |
|---|---|---|---:|---:|---:|---:|---:|---:|
| foldA | K-35-51-B-a | `fixed` | 128 | 0.875 | 0.586 | 0.6820 | 0.109 | 60 |
| foldA | K-35-51-B-a | `detector` | 128 | 0.883 | 0.570 | 0.6776 | 0.109 | 60 |
| foldB | K-35-8-G-a | `fixed` | 34 | 1.000 | 0.529 | 0.7482 | 0.000 | 0 |
| foldB | K-35-8-G-a | `detector` | 34 | 1.000 | 0.471 | 0.7455 | 0.000 | 0 |
| foldC | K-34-35-B-g | `fixed` | 7 | 1.000 | 0.857 | 0.7717 | 0.000 | 3 |
| foldC | K-34-35-B-g | `detector` | 7 | 1.000 | 0.571 | 0.7779 | 0.000 | 3 |

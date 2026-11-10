# RTX 4090 / Q4_K: Vor-Merge-Prüfung, 10. November 2026

Geprüft wurde `origin/chore/sync-upstream-20261107`, Commit `07f38482`, in einem separaten Worktree. Die Messungen erfolgten vor der anschließenden Versionierung dieser Belege; der ursprüngliche Checkout ist unverändert. Der veröffentlichte Prüfbranch enthält inzwischen `dc61f864`; dessen spätere Änderungen sind durch diese Messungen nicht qualifiziert. Es wurde kein Merge durchgeführt. Die angefragten Korrektheits-, Sanitizer-, Benchmark- und Modellregressionsläufe sind durchgeführt. Eine uneingeschränkte Merge-Empfehlung ergibt sich daraus nicht: Es gibt formabhängige Q4_K-Verlangsamungen und einen offenen Sanitizer-/Watchdog-Befund.

## Bedingungen und Vergleichsgrenzen

RTX 4090, 24 GB, Windows/WDDM mit aktivem Desktop, Treiber 621.14, nvcc 13.4.92, MSVC 19.44.35229, native SM89. Compute Sanitizer 2026.3. Beide Engine-Binaries melden Runtime und Driver API 13.5. Das ist unter Windows ab CUDA 13 laut dem installierten `cuda_runtime_api.h` möglich, weil die Runtime mit dem Displaytreiber geliefert wird; kompiliert wurde mit dem 13.4-Toolkit.

Der Op-Vergleich verwendet dieselbe Kandidaten-Binärdatei und dieselben Eingaben; nur `gguf_matrix/q4_k` wechselt zwischen kompiliertem Altpfad und `async`. Der Engine-Vergleich verwendet den vorhandenen Produktivbuild `g237cdff9` und den neuen Build `g07f38482`. Er erfasst das gesamte Upstream-Update und isoliert den Q4_K-Anteil nicht.

Modell: `C:/ninfer-all/models/Huihui-Qwen3.8-27B-GSQ-RCO-IQ3S-Vision-DFlash2.ninfer`. Dieses gemischte IQ3_S-Artefakt enthält laut Loader auch Q4_K. Ein zusätzliches dediziertes Q4_K-Gesamtmodell wurde nicht getestet; es wurde nichts heruntergeladen oder konvertiert.

Nur Qualifikationscode wurde im separaten Worktree ergänzt: unabhängiger Q4_K-Hostdecoder, FP64-MoE-Oracle, gezielter Racecheck, Alignment-Einstieg und explizite Routenauswahl im Benchmark. Keine Produktionsimplementierung wurde geändert. Der vollständige Patch liegt in `qualification-harness.patch`.

## Korrektheit und Sanitizer

- Dense: öffentlicher Linear/LinearAdd/LinearSwiGLU-Pfad gegen naives FP64-Produkt unabhängig decodierter gespeicherter Gewichte, BF16-Eingaben, beide Matrixrouten; bestanden. Bestehende Grenzen: relatives RMS <2 %, normiertes Maximum <8 %, für SwiGLU doppelte Budgets wegen zweier quantisierter Produkte. Die Q8-Aktivierungsquantisierung bleibt Teil der Op-Semantik.
- Alt/Async: bitgleiche FP32-/BF16-Ausgaben einschließlich Addieren und Stream-K-Zwischen-/Fixupdaten, drei Formen `(N,K)=(128,5120),(17408,5120),(5120,17408)`, T=64,65,128,129,256,1280,4096; bestanden, mit und ohne Fixup. Die Geometrien prüfen auch die 64/65- und 128/129-Grenzen.
- MoE: bitgleiche Alt-/Async-Produkte bei 1024,4096,8192 gerouteten Paaren. Unabhängiger FP64-Oracle bei 1024 Paaren, N=640,K=2560, acht Experten, BF16-Eingaben, beide Routen direkt geprüft: RMS 0,560962 %, normiertes Maximum 0,466135 %; bestanden.
- Vollständiger Expert-Delivery-Test: bestanden, einschließlich neuer Bank-/Cache-/Leihslot-Ausrichtung.
- Vollständiger MMQ-Memcheck und -Synccheck: Exit 0, jeweils 0 Fehler.
- Gezielter Racecheck: Exit 0, 0 Fehler/0 Warnungen, 153,60 s. Sechs Dense-Fälle mit Stream-K-Fixup (drei Formen, T=65,129) und MoE mit 1024 Paaren. Der vollständige Racecheck wurde nach 350,08 s ohne Abschlussbericht wegen seiner Kosten abgebrochen; er ist nicht qualifiziert. Der gezielte Lauf ersetzt diese vollständige Abdeckung nicht.
- Neue Expert-Alignment-Prüfung allein unter Memcheck: Exit 0, 0 Fehler. Der vollständige Expert-Delivery-Memcheck endete zuvor nach 121,72 s im 120-s-CUDA-Watchdog (`a stream`, kein antwortender Host-Service). Memcheck zeigte dabei 0 Speicherfehler, der Prozess scheiterte aber. Ursache und Vereinbarkeit dieses Gesamtpfads mit Instrumentierung sind ungeklärt; kein erfolgreicher Gesamtlauf wird behauptet.
- Reales DFlash2: K=5, Batch=1, Graphs und optimierter Pfad an, INT8-KV, Vision an, ein State-Slot; bestanden, 18/20 Draft-Tokens akzeptiert, State-D2H=4/H2D=2.

Die ersten Bundle-Starts scheiterten mit `0xC0000135`, weil beim gezielten Build der separate DLL-Staging-Target fehlte. `ninfer_stage_test_runtime_dlls` wurde nachgebaut; die betroffenen Tests wurden erfolgreich wiederholt. Alle ersten Fehlstarts bleiben in `checks.json` erhalten. Der uneingeschränkt parallele Build verursachte RAM-Druck unter 200 MiB frei und wurde beendet. Fortsetzung erst mit vier, nach Speicherbeobachtung mit acht Jobs; der Build war erfolgreich. Das sind behobene Ausführungsprobleme, keine bestandenen Erstläufe.

## Q4_K-Op-Leistung

Öffentlicher Linear-Op, BF16-Ausgabe, eager, drei Warmups, 20 kalte Messungen je Prozess, 256 MiB L2-Flush. Zwei A/B-Paare mit umgekehrter Reihenfolge. Werte unten sind die beiden gemessenen Medianlatenzen sowie der Median ihrer paarweisen Änderung; negatives Delta bedeutet schneller. Dies ist ein Op-Ergebnis, kein vollständiger Q4_K-Modellbenchmark.

Auf den 18 Punkten oberhalb T=64: 11 schnellere, 6 langsamere und ein gleicher Median. Größte Medianverbesserung -9,48 %, größte Verschlechterung +6,25 %. Die drei T=64-Kontrollen wählen denselben Kernel; zwei sind gleich, eine liegt bei -0,51 %. Keine Verschlechterung wird als Rauschen verworfen. Bei N=128,T=129 steigen beide Messungen von 32,768 auf 34,816 us (+2,048 us/+6,25 %). Bei N=5120,K=17408,T=65 widersprechen sich die Paarergebnisse (+3,36 % und -0,80 %); der Median ist +1,28 %. Bei derselben Form mit T=4096 reichen die Paarergebnisse von -2,68 bis -8,86 %.

| N | K | T | Alt us (A/B-Paare) | Async us | Latenzänderung % (Paare) | Median % |
|---:|---:|---:|---|---|---|---:|
| 128 | 5120 | 64 | 29.696, 29.696 | 29.696, 29.696 | 0.000, 0.000 | +0.000 |
| 128 | 5120 | 65 | 35.872, 35.840 | 37.888, 37.888 | 5.620, 5.714 | +5.667 |
| 128 | 5120 | 128 | 37.888, 38.912 | 39.936, 38.976 | 5.405, 0.164 | +2.785 |
| 128 | 5120 | 129 | 32.768, 32.768 | 34.816, 34.816 | 6.250, 6.250 | +6.250 |
| 128 | 5120 | 512 | 35.840, 35.840 | 37.888, 37.952 | 5.714, 5.893 | +5.804 |
| 128 | 5120 | 2048 | 57.344, 57.344 | 58.368, 58.368 | 1.786, 1.786 | +1.786 |
| 128 | 5120 | 4096 | 87.040, 87.040 | 87.040, 87.040 | 0.000, 0.000 | +0.000 |
| 17408 | 5120 | 64 | 88.064, 88.064 | 88.064, 88.064 | 0.000, 0.000 | +0.000 |
| 17408 | 5120 | 65 | 128.000, 128.000 | 123.904, 122.880 | -3.200, -4.000 | -3.600 |
| 17408 | 5120 | 128 | 133.120, 133.120 | 124.928, 122.880 | -6.154, -7.692 | -6.923 |
| 17408 | 5120 | 129 | 222.208, 222.208 | 211.968, 212.992 | -4.608, -4.147 | -4.378 |
| 17408 | 5120 | 512 | 423.936, 422.912 | 398.336, 397.312 | -6.039, -6.053 | -6.046 |
| 17408 | 5120 | 2048 | 1608.640, 1608.700 | 1475.580, 1475.580 | -8.272, -8.275 | -8.273 |
| 17408 | 5120 | 4096 | 3192.830, 3132.420 | 2921.470, 2888.700 | -8.499, -7.781 | -8.140 |
| 5120 | 17408 | 64 | 87.040, 87.040 | 87.040, 86.144 | 0.000, -1.029 | -0.515 |
| 5120 | 17408 | 65 | 122.016, 128.000 | 126.112, 126.976 | 3.357, -0.800 | +1.278 |
| 5120 | 17408 | 128 | 134.144, 133.120 | 124.928, 124.928 | -6.870, -6.154 | -6.512 |
| 5120 | 17408 | 129 | 223.232, 223.232 | 211.968, 210.944 | -5.046, -5.505 | -5.275 |
| 5120 | 17408 | 512 | 427.008, 427.008 | 397.312, 396.288 | -6.954, -7.194 | -7.074 |
| 5120 | 17408 | 2048 | 1637.380, 1635.330 | 1481.730, 1480.700 | -9.506, -9.456 | -9.481 |
| 5120 | 17408 | 4096 | 3284.000, 3258.370 | 3195.900, 2969.600 | -2.683, -8.862 | -5.773 |

## Engine-Leistung, Ressourcen und Kurzzeitstabilität

Plain und DFlash2-5, identische Gewichte, Korpus, rk4v4-KV, FP16-GDN, Chunk1024, Graphs, Kontextkapazität16384. Pro Modus drei wechselnde A/B-Paare, je Prozess ein Warmup und zwei Messungen pro Fall. Fälle: pp512, pp2048, tg128, pp512+tg128, pp2048+tg128, pp8192+tg128. Zwölf Prozesse und 144 gemessene Generierungen: alle beendet, alle erwarteten Tokenbudgets vollständig. Kein Langzeit- oder allgemeiner Nebenläufigkeits-Stresstest.

Die Tabelle zeigt je Fall den Median der drei paarweisen Prozentänderungen und deren gesamte Spannweite. Beim Durchsatz ist positiv besser, bei TTFT negativ. Keine statistische Signifikanz oder Ursache kleiner Änderungen wird behauptet.

| Modus | Fall | Prefill t/s Änderung %: Median [min,max] | Decode t/s Änderung % | TTFT Änderung % |
|---|---|---|---|---|
| plain | pp512 | -0.150 [-0.791,+0.570] | — | +0.157 [-0.557,+0.797] |
| plain | pp2048 | +0.301 [-0.164,+0.460] | — | -0.299 [-0.458,+0.160] |
| plain | tg128 | — | -0.023 [-0.238,-0.021] | -1.647 [-3.421,+1.565] |
| plain | pp512+tg128 | -0.112 [-0.532,+0.704] | -0.023 [-0.074,+0.007] | +0.109 [-0.694,+0.534] |
| plain | pp2048+tg128 | +0.448 [-0.050,+0.628] | -0.038 [-0.079,+0.037] | -0.448 [-0.628,+0.049] |
| plain | pp8192+tg128 | +0.075 [-0.103,+0.316] | +0.035 [-0.122,+0.074] | -0.074 [-0.314,+0.103] |
| dflash2 | pp512 | +0.488 [+0.376,+1.029] | — | -0.480 [-1.041,-0.379] |
| dflash2 | pp2048 | +0.442 [+0.273,+0.501] | — | -0.445 [-0.500,-0.268] |
| dflash2 | tg128 | — | -0.155 [-0.343,+0.096] | +0.376 [+0.149,+2.137] |
| dflash2 | pp512+tg128 | +0.443 [+0.387,+0.586] | -0.028 [-0.083,+0.129] | -0.443 [-0.576,-0.387] |
| dflash2 | pp2048+tg128 | +0.380 [+0.292,+0.496] | -0.074 [-0.170,-0.024] | -0.379 [-0.493,-0.289] |
| dflash2 | pp8192+tg128 | +0.359 [+0.285,+0.484] | +0.269 [+0.269,+0.417] | -0.357 [-0.480,-0.284] |

Plain-Prefill-Mediane liegen zwischen -0,150 und +0,448 %, DFlash2-Prefill zwischen +0,359 und +0,488 %. Decode-Mediane liegen bei Plain zwischen -0,038 und +0,035 %, bei DFlash2 zwischen -0,155 und +0,269 %. Alle Einzelpaare zusammen: Prefill -0,791 bis +1,029 %, Decode -0,343 bis +0,417 %, TTFT -3,421 bis +2,137 %. Die gesamte Requestdauer reicht im Median je Fall von -0,480 bis +0,179 %, über alle Paare von -1,041 bis +0,797 %. Es gibt damit sowohl kleine Verbesserungen als auch Verschlechterungen; kein pauschaler Engine-Gewinn wird behauptet. Insbesondere tg128-Plain-Decode fällt in allen drei Paaren leicht ab; DFlash2 pp2048+tg128-Decode ebenso.

Prozess-VRAM-Peaks (WDDM-PDH, ungefähr 0,25-s-Sampling plus nvidia-smi-Aufrufzeit) sind in allen sechs Prozessen pro Modus exakt gleich: Plain 12240,242 MiB, DFlash2 14876,266 MiB. Das schließt kürzere, nicht erfasste Spitzen nicht aus. Gesamt-GPU-Peaks einschließlich Desktop: Plain Basis [13777,13768,13768], Kandidat [13773,13768,13768] MiB; DFlash2 Basis [16400,16404,16413], Kandidat [16404,16404,16414] MiB. Damit liegen die DFlash2-Kartenpeaks im Kandidaten um [4,0,1] MiB höher; eine Zuordnung dieses Desktop-weiten Unterschieds zum Engine ist nicht belegt.

plain: gemeldete Reservierungen gleich: Gewichte 11755300864 B, Sequenz 388160000 B, Workspace 305356032 B, Runtime 706098944 B. Workspace-Allokator-Peaks in allen sechs Prozessen: [305356032] B.

dflash2: gemeldete Reservierungen gleich: Gewichte 14360105472 B, Sequenz 511781376 B, Workspace 305356032 B, Runtime 1219790592 B. Workspace-Allokator-Peaks in allen sechs Prozessen: [305356032] B.

GPU-Takte, Temperaturen und Leistungen wurden mitgesampelt; kein Clock-Lock, aktiver WDDM-Desktop. Die Rohdaten enthalten jeden Prozess und jede Wiederholung. Ressourcen- und Leistungsbehauptungen gelten nur für diese Konfiguration. Shared-Memory-Belegung, andere GPUs/Toolchains, andere Familien und ein dediziertes Q4_K-Modell wurden nicht neu vermessen.

## Modellregression und Disposition

Drei Prompts (68,68,3585 Eingabetokens), je Plain und DFlash2, je64 greedy Ausgabetokens, Kontext8192, rk4v4, FP16-GDN und Chunk1024. In allen sechs Fällen stimmen Token-IDs, Text und Promptlänge zwischen dem vorhandenen Produktivbuild und dem Kandidaten exakt überein. Plain und DFlash2 unterscheiden sich dagegen bereits in der Basis bei allen drei Prompts: erste unterschiedliche Tokens41,32,8; im Kandidaten genau dieselben Unterschiede. Das ist keine neu beobachtete Drift und keine Bestätigung einer exakten Plain-/DFlash2-Äquivalenz.

Numerische Q4_K-Prüfungen und der kurze Produktivmodell-Regressionslauf sind bestanden. Die gemessenen Op-Verlangsamungen, der ungeklärte vollständige Expert-Delivery-Memcheck und die unvollständige vollständige Racecheck-Abdeckung bleiben sichtbar. Vor einer uneingeschränkten Merge-Freigabe sollten die formabhängige Async-Routenwahl und der Watchdog-/Instrumentierungspfad geklärt werden. Diese Prüfung ändert keine Produktionsroute und implementiert keinen Fix.

Befehle, Umgebungsvariablen und Exitcodes: `checks.json`, `kernel-bench/records.json`, `engine-ab/*.process.json`, `quality/summary.json`; alle Logs bleiben daneben erhalten. `engine-summary.json` und `kernel-bench/summary.json` enthalten die vollständigen Vergleiche.

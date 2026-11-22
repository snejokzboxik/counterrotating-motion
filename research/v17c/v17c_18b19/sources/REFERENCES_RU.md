# Источники физических данных

## Заряд и удержание заряда

1. M. Pauthenier, M. Moreau-Hanot, “La charge des particules sphériques dans un champ ionisé,” *Journal de Physique et le Radium* 3 (1932), [DOI: 10.1051/jphysrad:01932003012059000](https://doi.org/10.1051/jphysrad:01932003012059000). Классическое field-charge saturation; требует фактической зарядной среды/поля.
2. G. I. Tardos, R. W. L. Snaddon, P. W. Dietz, “Electrical Charge Measurements on Fine Airborne Particles,” *IEEE Transactions on Industry Applications* 20 (1984), [DOI: 10.1109/TIA.1984.4504644](https://doi.org/10.1109/TIA.1984.4504644). Первичные corona-charging измерения polystyrene latex и Faraday-cage сравнение; размеры исследования существенно меньше 70–90 μm.
3. J. E. J. Dalley et al., “Measurement of charge of airborne 3–10 μm spherical dielectric particles charged in an AC unipolar charger,” *Journal of Aerosol Science* (2005), [institutional record](https://uhra.herts.ac.uk/id/eprint/24256/). Индивидуальные charge distributions; использовать как методический, а не количественно прямой перенос.
4. J. Méndez Harper et al., “The lifetime of charged dust in the atmosphere,” *PNAS Nexus* 1 (2022), [DOI: 10.1093/pnasnexus/pgac220](https://doi.org/10.1093/pnasnexus/pgac220). Charge lifetime в акустически левитируемых частицах; не RF-trap test.

## Диэлектрические материалы

5. Bangs Laboratories, *Material Properties of Polystyrene and PMMA Microspheres*, [technical note](https://bangslabs.com/wp-content/uploads/BL_TSD0021_Polymer_MaterialProp.pdf). Bulk PS permittivity/strength reference; не измерение charged microsphere surface в воздухе.
6. M. A. Saucedo-Espinosa et al., “Polarization behavior of polystyrene particles under direct current and low-frequency (<1 kHz) electric fields in dielectrophoretic systems,” *Electrophoresis* 37 (2016), [DOI: 10.1002/elps.201500338](https://doi.org/10.1002/elps.201500338). Низкочастотная поляризация nano-PS в жидкой среде; не переносить комплексный коэффициент напрямую на 35–45 μm spheres in air.
7. M. Tirado, C. Grosse, “Dependence of the broad frequency dielectric spectra of colloidal polystyrene particle suspensions on the difference between the counterion and co-ion diffusion coefficients,” *JCIS* 298 (2006), [DOI: 10.1016/j.jcis.2006.01.027](https://doi.org/10.1016/j.jcis.2006.01.027). Комплексные spectra latex/electrolyte suspension; свидетельствует о surface/media dependence, но не даёт нужный P2 параметр для сухого bead.
8. 3M, *Glass Bubbles K, S and iM Series Product Information*, [official technical PDF](https://multimedia.3m.com/mws/media/91049O/3m-glass-bubbles-k-s-and-im-series.pdf?fn=GlassMicrospheresKS_Series_Celum). Grade ranges true density, size and estimated dielectric data. K1 median size and density are distributions/specifications, не точные свойства единичного экземпляра.

## Пробой и корона

9. P. Bas-Calopa, J.-R. Riba, M. Moreno-Eguilaz, “Measurement of Corona Discharges under Variable Geometry, Frequency and Pressure Environment,” *Applied Sciences* 12 (2022), [PMC](https://pmc.ncbi.nlm.nih.gov/articles/PMC8915065/). Первичные corona experiments; показывает geometry/frequency dependence.
10. T. Luo, M. Ghaffarian Niasar, P. T. M. Vaessen, “Experimental Study of Air Discharge Under Various Frequency Stress with Inhomogeneous Field,” ISH 2023, [DOI: 10.1049/icp.2024.0779](https://doi.org/10.1049/icp.2024.0779). Point-plane 50 Hz–5 kHz tests; близкая частотная область, другая форма электродов/зазоры.
11. J.-R. Riba, M. Soltany, “Modified Paschen curves for non-uniform fields in air based on a rod–plane electrode geometry,” *Plasma Sources Sci. Technol.* (2025), [DOI: 10.1088/1361-6595/ae23fa](https://doi.org/10.1088/1361-6595/ae23fa). Первичные данные/модель для rod-plane; transfer к 24 rods and two-tone field требует отдельной валидации.
12. J.-R. Riba, M. Soltany, “A dataset of corona inception and complete breakdown for sub-atmospheric air under alternating and direct current supply,” *Scientific Data* 13 (2026), [DOI: 10.1038/s41597-026-07721-2](https://doi.org/10.1038/s41597-026-07721-2). 1764 tests across pressure/gap and 50Hz AC/DC; не покрывает конкретный rod-shield gap и два независимых RF tone.

## Внутренние исходные исследования

13. V17C.12B/V16 MFS 24 rods: `h_{+2}=0.8703168911`, `h_{-4}=0.7982172624`; geometric model is 2D infinite rods and grounded circular shield. Source pin `data/source_benchmarks.json`.
14. V17C.16 report and source code: dimensional PS input, MFS Oseen/Coulomb trajectories, P1 εr=2.6 sensitivity and 2D MFS surface-field screen; commit `6708ec62acec19f11ade7b8bdc3d24c0b6985a33`.
15. V17C.17 `TWO_PARTICLE_COUNTERROTATION_DYNAMICS_RU.md`, configuration and raw results: phys0570 and scaled refine12 long continuation; commit `718d9eadf5873cdf1a5b602de31ec20647092f0f`.

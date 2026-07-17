from __future__ import annotations

from dataclasses import dataclass

import numpy as np


PA_TO_PSF = 0.020885434273039
REF_PRESSURE_PA = 20.0e-6
REF_TIME_S = 0.07


@dataclass(frozen=True)
class PLdBResult:
    pldb: float
    total_loudness_sones: float
    max_sones: float
    summation_factor: float
    band_centers_hz: np.ndarray
    band_spl_db: np.ndarray
    equivalent_loudness_db: np.ndarray
    sones: np.ndarray
    band_energy_pa2: np.ndarray
    method: str
    pad_front_multiplier: int
    pad_rear_multiplier: int
    len_window_points: int
    sample_rate_hz: float | None = None
    nyquist_hz: float | None = None
    fft_length: int | None = None
    fft_duration_s: float | None = None
    event_correction_db: float = 0.0
    band_supported_fraction: np.ndarray | None = None


def third_octave_bands_bolander_2019() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return Table 1 one-third-octave bands from Bolander et al. 2019."""

    centers = np.array(
        [
            1.0,
            1.25,
            1.6,
            2.0,
            2.5,
            3.15,
            4.0,
            5.0,
            6.3,
            8.0,
            10.0,
            12.5,
            16.0,
            20.0,
            25.0,
            31.5,
            40.0,
            50.0,
            63.0,
            80.0,
            100.0,
            125.0,
            160.0,
            200.0,
            250.0,
            315.0,
            400.0,
            500.0,
            630.0,
            800.0,
            1000.0,
            1250.0,
            1600.0,
            2000.0,
            2500.0,
            3150.0,
            4000.0,
            5000.0,
            6300.0,
            8000.0,
            10000.0,
            12500.0,
        ],
        dtype=float,
    )
    lower = np.array(
        [
            0.89,
            1.12,
            1.41,
            1.78,
            2.24,
            2.82,
            3.55,
            4.47,
            5.62,
            7.08,
            8.91,
            11.2,
            14.1,
            17.8,
            22.4,
            28.2,
            35.5,
            44.7,
            56.2,
            70.8,
            89.1,
            112.0,
            141.0,
            178.0,
            224.0,
            282.0,
            355.0,
            447.0,
            562.0,
            708.0,
            891.0,
            1120.0,
            1410.0,
            1780.0,
            2240.0,
            2820.0,
            3550.0,
            4470.0,
            5620.0,
            7080.0,
            8910.0,
            11200.0,
        ],
        dtype=float,
    )
    upper = np.array(
        [
            1.12,
            1.41,
            1.78,
            2.24,
            2.82,
            3.55,
            4.47,
            5.62,
            7.08,
            8.91,
            11.2,
            14.1,
            17.8,
            22.4,
            28.2,
            35.5,
            44.7,
            56.2,
            70.8,
            89.1,
            112.0,
            141.0,
            178.0,
            224.0,
            282.0,
            355.0,
            447.0,
            562.0,
            708.0,
            891.0,
            1120.0,
            1410.0,
            1780.0,
            2240.0,
            2820.0,
            3550.0,
            4470.0,
            5620.0,
            7080.0,
            8910.0,
            11200.0,
            14100.0,
        ],
        dtype=float,
    )
    return centers, lower, upper


def third_octave_bands_nasa_2025() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return the exact 1.25 Hz--20 kHz analysis bands in NASA/TM-20250003346.

    NASA defines 43 base-10 one-third-octave bands using ``fr = 1000 Hz``,
    ``G = 10**(3/10)``, and band indices ``o = 1..43``.  These are exact
    band definitions; the familiar nominal labels (1.25, 1.6, ..., 20,000)
    are only rounded display values.
    """

    reference_hz = 1000.0
    octave_ratio = 10.0 ** (3.0 / 10.0)
    band_index = np.arange(1.0, 44.0)
    centers = reference_hz * octave_ratio ** ((band_index - 30.0) / 3.0)
    lower = centers * octave_ratio ** (-1.0 / 6.0)
    upper = centers * octave_ratio ** (+1.0 / 6.0)
    return centers, lower, upper


def summation_factor_table_bolander_2019() -> tuple[np.ndarray, np.ndarray]:
    """Return Table B.2 max-sones to summation-factor data."""

    max_sones = np.array(
        [
            0.181,
            0.196,
            0.212,
            0.230,
            0.248,
            0.269,
            0.290,
            0.314,
            0.339,
            0.367,
            0.396,
            0.428,
            0.463,
            0.500,
            0.540,
            0.583,
            0.630,
            0.680,
            0.735,
            0.794,
            0.857,
            0.926,
            1.00,
            1.08,
            1.17,
            1.26,
            1.36,
            1.47,
            1.59,
            1.72,
            1.85,
            2.00,
            2.16,
            2.33,
            2.52,
            2.72,
            2.94,
            3.18,
            3.43,
            3.70,
            4.00,
            4.32,
            4.67,
            5.04,
            5.44,
            5.88,
            6.35,
            6.86,
            7.41,
            8.00,
            8.64,
            9.33,
            10.1,
            10.9,
            11.8,
            12.7,
            13.7,
            14.8,
            16.0,
            17.3,
            18.7,
            20.2,
            21.8,
            23.5,
            25.4,
            27.4,
            29.6,
            32.0,
            34.6,
            37.3,
            40.3,
            43.5,
            47.0,
            50.8,
            54.9,
            59.3,
            64.0,
            69.1,
            74.7,
            80.6,
            87.1,
            94.1,
            102.0,
            110.0,
            119.0,
            128.0,
            138.0,
            149.0,
            161.0,
            174.0,
            188.0,
            203.0,
            219.0,
            237.0,
            256.0,
        ],
        dtype=float,
    )
    factor = np.array(
        [
            0.100,
            0.122,
            0.140,
            0.158,
            0.174,
            0.187,
            0.200,
            0.212,
            0.222,
            0.232,
            0.241,
            0.250,
            0.259,
            0.267,
            0.274,
            0.281,
            0.287,
            0.293,
            0.298,
            0.303,
            0.308,
            0.312,
            0.316,
            0.319,
            0.320,
            0.322,
            0.322,
            0.320,
            0.319,
            0.317,
            0.314,
            0.311,
            0.308,
            0.304,
            0.300,
            0.296,
            0.292,
            0.288,
            0.284,
            0.279,
            0.275,
            0.270,
            0.266,
            0.262,
            0.258,
            0.253,
            0.248,
            0.244,
            0.240,
            0.235,
            0.230,
            0.226,
            0.222,
            0.217,
            0.212,
            0.208,
            0.204,
            0.200,
            0.197,
            0.195,
            0.194,
            0.193,
            0.192,
            0.191,
            0.190,
            0.190,
            0.190,
            0.190,
            0.190,
            0.190,
            0.191,
            0.191,
            0.192,
            0.193,
            0.194,
            0.195,
            0.197,
            0.199,
            0.201,
            0.203,
            0.205,
            0.208,
            0.210,
            0.212,
            0.215,
            0.217,
            0.219,
            0.221,
            0.223,
            0.224,
            0.225,
            0.226,
            0.227,
            0.227,
            0.227,
        ],
        dtype=float,
    )
    return max_sones, factor


def equivalent_loudness_to_sones_table_bolander_2019() -> tuple[np.ndarray, np.ndarray]:
    """Return Table B.1 equivalent-loudness to sones values.

    The table is represented at 1 dB intervals from 1 to 140 dB. Values follow
    the Mark VII sone power law at the table points and are linearly
    interpolated, matching the PyLdB / Bolander 2019 implementation path.
    """

    l_eq = np.arange(1.0, 141.0, dtype=float)
    sones = np.power(2.0, (l_eq - 32.0) / 9.0)
    return l_eq, sones


def equivalent_loudness_mark_vii(levels_db: np.ndarray, centers_hz: np.ndarray) -> np.ndarray:
    """Transform one-third-octave SPLs to 3150 Hz equivalent loudness."""

    levels = np.asarray(levels_db, dtype=float)
    centers = np.asarray(centers_hz, dtype=float)
    l_eq = np.zeros_like(levels)
    for i, level in enumerate(levels):
        if i > 39:
            l_eq[i] = level + 4.0 * (39.0 - i)
        elif 35 <= i <= 39:
            l_eq[i] = level
        elif 32 <= i <= 34:
            l_eq[i] = level - 2.0 * (35.0 - i)
        elif 26 < i <= 31:
            l_eq[i] = level - 8.0
        elif 20 <= i <= 26:
            lower = 76.0 + 1.5 * (26 - i)
            upper = 121.0 + 1.5 * (26 - i)
            x_value = 1.5 * (26 - i)
            l_eq[i] = _loud_limits_400(float(centers[i]), lower, upper, float(level), x_value)
        else:
            if centers[i] <= 1.0:
                b_value = -np.inf
            else:
                b_value = 160.0 - ((160.0 - level) * np.log10(80.0)) / np.log10(centers[i])
            l_eq[i] = _loud_limits_400(80.0, 86.5, 131.5, float(b_value), 10.5)
    return l_eq


def sones_from_equivalent_loudness(l_eq_db: np.ndarray) -> np.ndarray:
    """Convert L_eq to sones using Table B.1 linear interpolation."""

    l_eq = np.asarray(l_eq_db, dtype=float)
    table_l_eq, table_sones = equivalent_loudness_to_sones_table_bolander_2019()
    return np.interp(l_eq, table_l_eq, table_sones, left=0.0, right=table_sones[-1])


def summation_factor_table_nasa_2025() -> tuple[np.ndarray, np.ndarray]:
    """Return the NASA 2025 low-level-complete Mark VII summation table."""

    legacy_sones, legacy_factor = summation_factor_table_bolander_2019()
    return (
        np.concatenate((np.array([0.0, 0.113]), legacy_sones)),
        np.concatenate((np.array([0.0, 0.0]), legacy_factor)),
    )


def equivalent_loudness_mark_vii_nasa_2025(
    levels_db: np.ndarray,
    centers_hz: np.ndarray,
) -> np.ndarray:
    """Apply the Jackson-Leventhall geometric equal-loudness construction.

    The legacy implementation includes a nominal 1 Hz band before the 1.25 Hz
    band.  NASA's exact 2025 band set begins at 1.25 Hz, so its array index is
    shifted by one while the original Jackson-Leventhall band numbering is
    retained.
    """

    levels = np.asarray(levels_db, dtype=float)
    centers = np.asarray(centers_hz, dtype=float)
    if levels.shape != centers.shape:
        raise ValueError("levels and band centers must have the same shape")
    equivalent = np.zeros_like(levels)
    for nasa_index, level in enumerate(levels):
        legacy_index = nasa_index + 1
        if legacy_index > 39:
            equivalent[nasa_index] = level + 4.0 * (39.0 - legacy_index)
        elif 35 <= legacy_index <= 39:
            equivalent[nasa_index] = level
        elif 32 <= legacy_index <= 34:
            equivalent[nasa_index] = level - 2.0 * (35.0 - legacy_index)
        elif 26 < legacy_index <= 31:
            equivalent[nasa_index] = level - 8.0
        elif 20 <= legacy_index <= 26:
            lower = 76.0 + 1.5 * (26 - legacy_index)
            upper = 121.0 + 1.5 * (26 - legacy_index)
            x_value = 1.5 * (26 - legacy_index)
            equivalent[nasa_index] = _loud_limits_400(
                float(centers[nasa_index]),
                lower,
                upper,
                float(level),
                x_value,
            )
        else:
            b_value = 160.0 - (
                (160.0 - level) * np.log10(80.0) / np.log10(centers[nasa_index])
            )
            equivalent[nasa_index] = _loud_limits_400(
                80.0,
                86.5,
                131.5,
                float(b_value),
                10.5,
            )
    return equivalent


def sones_from_level_nasa_2025(level_db: np.ndarray | float) -> np.ndarray:
    """Convert loudness level in dB to sones using NASA's level-aware equations."""

    level = np.asarray(level_db, dtype=float)
    sones = np.zeros_like(level)
    high = level >= 32.0
    middle = (level > -3.0) & ~high
    sones[high] = np.power(2.0, (level[high] - 32.0) / 9.0)
    numerator = np.power(10.0, level[middle] / 10.0) - np.power(10.0, -0.3)
    denominator = np.power(10.0, 3.2) - np.power(10.0, -0.3)
    sones[middle] = np.cbrt(np.maximum(numerator / denominator, 0.0))
    return sones


def level_from_sones_nasa_2025(sones: np.ndarray | float) -> np.ndarray:
    """Convert sones to loudness level in dB using NASA's level-aware equations."""

    amplitude = np.asarray(sones, dtype=float)
    if np.any(amplitude < 0.0):
        raise ValueError("loudness amplitude cannot be negative")
    level = np.full_like(amplitude, -3.0)
    high = amplitude >= 1.0
    middle = (amplitude > 0.0) & ~high
    level[high] = 32.0 + 9.0 * np.log2(amplitude[high])
    low_argument = (
        (np.power(10.0, 3.2) - np.power(10.0, -0.3)) * amplitude[middle] ** 3
        + np.power(10.0, -0.3)
    )
    level[middle] = 10.0 * np.log10(low_argument)
    return level


def total_loudness_nasa_2025(sones: np.ndarray) -> tuple[float, float, float, float]:
    """Sum a Mark VII loudness spectrum and return PL, total, maximum, and F."""

    loudness = np.asarray(sones, dtype=float)
    if loudness.size == 0 or np.any(~np.isfinite(loudness)) or np.any(loudness < 0.0):
        raise ValueError("loudness spectrum must be finite, non-negative, and non-empty")
    max_sones = float(np.max(loudness))
    table_sones, table_factor = summation_factor_table_nasa_2025()
    summation_factor = float(
        np.interp(
            max_sones,
            table_sones,
            table_factor,
            left=table_factor[0],
            right=table_factor[-1],
        )
    )
    total_sones = max_sones + summation_factor * (float(np.sum(loudness)) - max_sones)
    pldb = float(level_from_sones_nasa_2025(total_sones))
    return pldb, float(total_sones), max_sones, summation_factor


def _uniform_waveform(time_s: np.ndarray, pressure_pa: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
    t = np.asarray(time_s, dtype=float)
    p = np.asarray(pressure_pa, dtype=float)
    finite = np.isfinite(t) & np.isfinite(p)
    t = t[finite]
    p = p[finite]
    if t.size < 8:
        raise ValueError("PLdB calculation requires at least 8 finite samples")
    order = np.argsort(t)
    t = t[order] - float(t[order][0])
    p = p[order].copy()
    differences = np.diff(t)
    dt = float(np.median(differences))
    if not np.isfinite(dt) or dt <= 0.0:
        raise ValueError("PLdB calculation requires a positive uniform time step")
    if not np.allclose(differences, dt, rtol=1.0e-6, atol=max(1.0e-12, dt * 1.0e-8)):
        raise ValueError("NASA narrow-band summation requires uniformly sampled time data")
    return t, p, dt


def stevens_mark_vii_pldb_nasa_2025(
    time_s: np.ndarray,
    pressure_pa: np.ndarray,
    *,
    pressure_scale: float = 1.0,
    time_scale: float = 1.0,
    taper_duration_s: float = 0.010,
    minimum_padded_duration_s: float = 2.0,
    event_correction_db: float = 0.0,
) -> PLdBResult:
    """Compute Mark VII PLdB using the methods recommended by NASA in 2025.

    This implementation follows NASA/TM-20250003346: exact base-10 bands from
    1.25 Hz to 20 kHz; a tapered waveform; power-of-two zero padding beyond
    two seconds; fractional apportionment of overlapping narrow-band bins;
    Jackson-Leventhall equal-loudness geometry; the level-aware sone equations;
    and the low-level-complete linear ``F(S)`` table.  No blanket ``-3 dB``
    adjustment is part of the NASA algorithm.  A nonzero event correction can
    be supplied explicitly for a separately justified measurement convention.
    """

    scaled_time = np.asarray(time_s, dtype=float) * float(time_scale)
    scaled_pressure = np.asarray(pressure_pa, dtype=float) * float(pressure_scale)
    _, pressure, dt = _uniform_waveform(scaled_time, scaled_pressure)

    taper_points = min(
        int(round(max(float(taper_duration_s), 0.0) / dt)),
        pressure.size // 4,
    )
    if taper_points > 0:
        taper = np.hanning(2 * taper_points)
        pressure[:taper_points] *= taper[:taper_points]
        pressure[-taper_points:] *= taper[taper_points:]

    minimum_samples = int(np.floor(max(float(minimum_padded_duration_s), 0.0) / dt)) + 2
    target_samples = max(int(pressure.size), minimum_samples)
    fft_length = 1 << (target_samples - 1).bit_length()
    pad_total = fft_length - pressure.size
    pad_front = pad_total // 2
    padded = np.pad(pressure, (pad_front, pad_total - pad_front))

    sample_rate = 1.0 / dt
    frequency = np.fft.rfftfreq(fft_length, d=dt)
    delta_f = sample_rate / fft_length
    transform = np.fft.rfft(padded)
    bin_energy = np.abs(transform) ** 2 * dt**2 * delta_f / REF_TIME_S
    if bin_energy.size > 2:
        bin_energy[1:-1] *= 2.0

    all_centers, all_lower, all_upper = third_octave_bands_nasa_2025()
    # Mark VII uses the 41 exact bands through 12.589 kHz (NASA TM Table 2).
    # The 16 and 20 kHz bands are part of the recommended general-purpose
    # one-third-octave analysis set, but are not inputs to the PL calculation.
    centers = all_centers[:41]
    lower = all_lower[:41]
    upper = all_upper[:41]
    bin_lower = frequency - 0.5 * delta_f
    bin_upper = frequency + 0.5 * delta_f
    band_energy = np.zeros_like(centers)
    for band_index, (band_lower, band_upper) in enumerate(zip(lower, upper)):
        overlap = np.maximum(
            0.0,
            np.minimum(bin_upper, band_upper) - np.maximum(bin_lower, band_lower),
        )
        band_energy[band_index] = float(np.sum(bin_energy * overlap / delta_f))

    nyquist = 0.5 * sample_rate
    represented_upper = nyquist + 0.5 * delta_f
    supported_width = np.maximum(0.0, np.minimum(upper, represented_upper) - lower)
    supported_fraction = np.clip(supported_width / (upper - lower), 0.0, 1.0)
    with np.errstate(divide="ignore", invalid="ignore"):
        spl = 10.0 * np.log10(band_energy / (REF_PRESSURE_PA**2))
    spl = np.where(np.isfinite(spl), spl + float(event_correction_db), -300.0)

    equivalent = equivalent_loudness_mark_vii_nasa_2025(spl, centers)
    sones = sones_from_level_nasa_2025(equivalent)
    pldb, total_sones, max_sones, summation_factor = total_loudness_nasa_2025(sones)
    return PLdBResult(
        pldb=pldb,
        total_loudness_sones=total_sones,
        max_sones=max_sones,
        summation_factor=summation_factor,
        band_centers_hz=centers,
        band_spl_db=spl,
        equivalent_loudness_db=equivalent,
        sones=sones,
        band_energy_pa2=band_energy,
        method="stevens_mark_vii_nasa_tm_20250003346_narrow_band_summation",
        pad_front_multiplier=0,
        pad_rear_multiplier=0,
        len_window_points=taper_points,
        sample_rate_hz=sample_rate,
        nyquist_hz=nyquist,
        fft_length=fft_length,
        fft_duration_s=fft_length * dt,
        event_correction_db=float(event_correction_db),
        band_supported_fraction=supported_fraction,
    )


def stevens_mark_vii_pldb(
    time_s: np.ndarray,
    pressure_pa: np.ndarray,
    pad_front_multiplier: int = 6,
    pad_rear_multiplier: int = 6,
    len_window_points: int = 800,
    pressure_scale: float = 1.0,
    time_scale: float = 1.0,
) -> PLdBResult:
    """Calculate Stevens Mark VII PLdB from pressure-time data.

    Implements the calculation path summarized by Bolander et al. 2019:
    Hanning window, zero padding, one-sided power spectrum, one-third-octave
    band energy, 3150 Hz equivalent loudness, sone conversion, Table B.2
    summation factor, and PLdB power-law conversion.
    """

    t = np.asarray(time_s, dtype=float)
    p = np.asarray(pressure_pa, dtype=float) * float(pressure_scale)
    finite = np.isfinite(t) & np.isfinite(p)
    t = t[finite]
    p = p[finite]
    if t.size < 8:
        raise ValueError("PLdB calculation requires at least 8 finite samples.")
    order = np.argsort(t)
    t = (t[order] - float(t[order][0])) * float(time_scale)
    p = p[order].copy()
    dt = float(np.median(np.diff(t)))
    if not np.isfinite(dt) or dt <= 0.0:
        raise ValueError("PLdB calculation requires a positive uniform time step.")

    n_window = min(int(len_window_points), max(int(p.size) // 3, 0))
    if n_window > 0:
        window = np.hanning(n_window * 2)
        p[:n_window] *= window[:n_window]
        p[-n_window:] *= window[n_window:]
    p = np.pad(p, (int(p.size * pad_front_multiplier), int(p.size * pad_rear_multiplier)))
    # Match the PyLdB / Bolander implementation convention: after padding, the
    # FFT spacing is duration / N rather than the raw sample spacing.
    dt_fft = dt * max((p.size - 1) / p.size, 1.0e-12)
    fft_values = np.fft.rfft(p)
    freq = np.fft.rfftfreq(p.size, d=dt_fft)
    power = np.abs(fft_values) ** 2 * dt_fft**2
    if power.size > 2:
        power[1:-1] *= 2.0

    centers, lower, upper = third_octave_bands_bolander_2019()
    interp_freq = np.append(lower, upper[-1])
    interp_power = np.interp(interp_freq, freq, power)
    full_freq = np.concatenate((freq, interp_freq))
    full_power = np.concatenate((power, interp_power))
    sort_idx = np.argsort(full_freq, kind="mergesort")
    full_freq = full_freq[sort_idx]
    full_power = full_power[sort_idx]

    band_energy = np.zeros_like(centers)
    for idx, (lo, hi) in enumerate(zip(lower, upper)):
        mask = np.nonzero((lo <= full_freq) & (full_freq <= hi))[0]
        if mask.size > 1:
            band_energy[idx] = np.trapezoid(full_power[mask], x=full_freq[mask])
    band_energy /= REF_TIME_S
    with np.errstate(divide="ignore", invalid="ignore"):
        spl = 10.0 * np.log10(band_energy / (REF_PRESSURE_PA**2)) - 3.0
    spl = np.where(np.isfinite(spl), spl, -300.0)

    l_eq = equivalent_loudness_mark_vii(spl, centers)
    sones = sones_from_equivalent_loudness(l_eq)
    max_sones = float(np.max(sones))
    table_sones, table_factor = summation_factor_table_bolander_2019()
    summation_factor = float(np.interp(max_sones, table_sones, table_factor, left=0.0, right=table_factor[-1]))
    total_sones = max_sones + summation_factor * (float(np.sum(sones)) - max_sones)
    pldb = float(32.0 + 9.0 * np.log2(max(total_sones, 1.0e-300)))

    return PLdBResult(
        pldb=pldb,
        total_loudness_sones=float(total_sones),
        max_sones=max_sones,
        summation_factor=summation_factor,
        band_centers_hz=centers,
        band_spl_db=spl,
        equivalent_loudness_db=l_eq,
        sones=sones,
        band_energy_pa2=band_energy,
        method="stevens_mark_vii_bolander_2019_tables",
        pad_front_multiplier=int(pad_front_multiplier),
        pad_rear_multiplier=int(pad_rear_multiplier),
        len_window_points=int(n_window),
    )


def _loud_limits_400(f_central: float, lower_limit: float, upper_limit: float, loudness: float, x_value: float) -> float:
    if loudness <= lower_limit:
        equivalent = 115.0 - ((115.0 - loudness) * np.log10(400.0)) / np.log10(f_central)
        return float(equivalent - 8.0)
    if loudness <= upper_limit:
        return float(loudness - x_value - 8.0)
    equivalent = 160.0 - ((160.0 - loudness) * np.log10(400.0)) / np.log10(f_central)
    return float(equivalent - 8.0)


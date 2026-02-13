"use client";

interface SpikeLogoProps {
  size?: number;
  className?: string;
}

export function SpikeLogo({ size = 36, className }: SpikeLogoProps) {
  // Unique IDs to prevent SVG gradient conflicts when multiple instances render
  const id = "spike-logo";

  return (
    <svg
      width={size}
      height={size}
      viewBox="0 0 512 512"
      fill="none"
      xmlns="http://www.w3.org/2000/svg"
      className={className}
      aria-label="Spike logo"
      role="img"
    >
      <defs>
        {/* Background gradient - deep, premium teal */}
        <linearGradient id={`${id}-bg`} x1="0%" y1="100%" x2="100%" y2="0%">
          <stop offset="0%" stopColor="#071F1C" />
          <stop offset="40%" stopColor="#0A2E2A" />
          <stop offset="100%" stopColor="#0D4A42" />
        </linearGradient>

        {/* Subtle surface shine for glass-like depth */}
        <linearGradient id={`${id}-shine`} x1="0%" y1="0%" x2="100%" y2="100%">
          <stop offset="0%" stopColor="white" stopOpacity="0.08" />
          <stop offset="50%" stopColor="white" stopOpacity="0.02" />
          <stop offset="100%" stopColor="white" stopOpacity="0" />
        </linearGradient>

        {/* Upper S-curve gradient - luminous accent green */}
        <linearGradient id={`${id}-curve-upper`} x1="0%" y1="0%" x2="100%" y2="100%">
          <stop offset="0%" stopColor="#A8E6CF" />
          <stop offset="50%" stopColor="#7DCEA0" />
          <stop offset="100%" stopColor="#4BA89A" />
        </linearGradient>

        {/* Lower S-curve gradient - deeper complementary teal */}
        <linearGradient id={`${id}-curve-lower`} x1="100%" y1="100%" x2="0%" y2="0%">
          <stop offset="0%" stopColor="#89D4B8" />
          <stop offset="50%" stopColor="#5FBAA8" />
          <stop offset="100%" stopColor="#3D9688" />
        </linearGradient>

        {/* Central aperture glow - the "intelligence" */}
        <radialGradient id={`${id}-glow`} cx="50%" cy="50%" r="18%" fx="50%" fy="50%">
          <stop offset="0%" stopColor="#D3E9D7" stopOpacity="0.9" />
          <stop offset="40%" stopColor="#7DCEA0" stopOpacity="0.4" />
          <stop offset="100%" stopColor="#4BA89A" stopOpacity="0" />
        </radialGradient>

        {/* Outer ambient glow */}
        <radialGradient id={`${id}-ambient`} cx="50%" cy="50%" r="45%" fx="50%" fy="50%">
          <stop offset="0%" stopColor="#7DCEA0" stopOpacity="0.12" />
          <stop offset="100%" stopColor="#7DCEA0" stopOpacity="0" />
        </radialGradient>

        {/* Edge highlight for upper curve */}
        <linearGradient id={`${id}-edge-upper`} x1="0%" y1="0%" x2="100%" y2="100%">
          <stop offset="0%" stopColor="white" stopOpacity="0.5" />
          <stop offset="100%" stopColor="white" stopOpacity="0.05" />
        </linearGradient>

        {/* Edge highlight for lower curve */}
        <linearGradient id={`${id}-edge-lower`} x1="100%" y1="100%" x2="0%" y2="0%">
          <stop offset="0%" stopColor="white" stopOpacity="0.35" />
          <stop offset="100%" stopColor="white" stopOpacity="0.05" />
        </linearGradient>
      </defs>

      {/* === BACKGROUND === */}
      {/* Rounded square - darker, more premium than before */}
      <rect width="512" height="512" rx="112" fill={`url(#${id}-bg)`} />

      {/* Glass surface shine */}
      <rect width="512" height="512" rx="112" fill={`url(#${id}-shine)`} />

      {/* Subtle ambient glow behind the mark */}
      <circle cx="256" cy="256" r="200" fill={`url(#${id}-ambient)`} />

      {/* === THE SENTINEL S MARK === */}
      {/*
        Two interlocking curves form an abstract "S" shape.
        Their negative space creates an aperture / eye at the center --
        the vigilant AI that never stops watching.
        The curves also echo a pulse waveform: the "spike" moment.
      */}

      {/* Upper curve - sweeps from upper-right, arcs left, tapers toward center */}
      {/* This is the dominant curve: bolder, brighter, leading the eye downward */}
      <path
        d={`
          M 392 196
          C 392 116, 292 80, 216 128
          C 140 176, 132 220, 220 256
          C 264 272, 288 256, 256 236
        `}
        stroke={`url(#${id}-curve-upper)`}
        strokeWidth="44"
        strokeLinecap="round"
        strokeLinejoin="round"
        fill="none"
      />

      {/* Upper curve - luminous edge highlight */}
      <path
        d={`
          M 392 196
          C 392 116, 292 80, 216 128
          C 140 176, 132 220, 220 256
          C 264 272, 288 256, 256 236
        `}
        stroke={`url(#${id}-edge-upper)`}
        strokeWidth="6"
        strokeLinecap="round"
        strokeLinejoin="round"
        fill="none"
      />

      {/* Lower curve - sweeps from lower-left, arcs right, tapers toward center */}
      {/* Complementary to upper: slightly thinner, creating asymmetric tension */}
      <path
        d={`
          M 120 316
          C 120 396, 220 432, 296 384
          C 372 336, 380 292, 292 256
          C 248 240, 224 256, 256 276
        `}
        stroke={`url(#${id}-curve-lower)`}
        strokeWidth="40"
        strokeLinecap="round"
        strokeLinejoin="round"
        fill="none"
      />

      {/* Lower curve - luminous edge highlight */}
      <path
        d={`
          M 120 316
          C 120 396, 220 432, 296 384
          C 372 336, 380 292, 292 256
          C 248 240, 224 256, 256 276
        `}
        stroke={`url(#${id}-edge-lower)`}
        strokeWidth="5"
        strokeLinecap="round"
        strokeLinejoin="round"
        fill="none"
      />

      {/* === CENTRAL APERTURE GLOW === */}
      {/* The "eye" - where the two curves create negative space */}
      {/* A soft radial glow suggests intelligence, awareness, focus */}
      <circle cx="256" cy="256" r="28" fill={`url(#${id}-glow)`} />

      {/* Inner focal point - the pupil of the sentinel */}
      <circle cx="256" cy="256" r="8" fill="#D3E9D7" opacity="0.85" />

      {/* Tiny specular highlight - adds life and sparkle */}
      <circle cx="252" cy="252" r="3" fill="white" opacity="0.7" />
    </svg>
  );
}

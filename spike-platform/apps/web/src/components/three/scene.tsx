"use client";

import { useEffect, useRef, useCallback } from "react";

interface Orb {
  x: number;
  y: number;
  vx: number;
  vy: number;
  radius: number;
  color: string;
  opacity: number;
  phase: number;
}

interface Particle {
  x: number;
  y: number;
  vx: number;
  vy: number;
  size: number;
  opacity: number;
  life: number;
  maxLife: number;
}

export function ThreeScene() {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const mouseRef = useRef({ x: 0.5, y: 0.5 });
  const orbsRef = useRef<Orb[]>([]);
  const particlesRef = useRef<Particle[]>([]);
  const frameRef = useRef<number>(0);
  const timeRef = useRef<number>(0);

  const initOrbs = useCallback((width: number, height: number) => {
    const colors = [
      "rgba(99, 140, 130,",  // sage
      "rgba(125, 206, 160,", // accent
      "rgba(211, 233, 215,", // mint
      "rgba(139, 175, 166,", // muted
    ];

    orbsRef.current = Array.from({ length: 5 }, (_, i) => ({
      x: Math.random() * width,
      y: Math.random() * height,
      vx: (Math.random() - 0.5) * 0.3,
      vy: (Math.random() - 0.5) * 0.3,
      radius: 150 + Math.random() * 200,
      color: colors[i % colors.length],
      opacity: 0.04 + Math.random() * 0.06,
      phase: Math.random() * Math.PI * 2,
    }));

    particlesRef.current = Array.from({ length: 80 }, () => ({
      x: Math.random() * width,
      y: Math.random() * height,
      vx: (Math.random() - 0.5) * 0.15,
      vy: (Math.random() - 0.5) * 0.15,
      size: 1 + Math.random() * 2,
      opacity: 0.1 + Math.random() * 0.4,
      life: Math.random() * 1000,
      maxLife: 800 + Math.random() * 400,
    }));
  }, []);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;

    const ctx = canvas.getContext("2d", { alpha: true });
    if (!ctx) return;

    let width = window.innerWidth;
    let height = window.innerHeight;

    const resize = () => {
      width = window.innerWidth;
      height = window.innerHeight;
      canvas.width = width * window.devicePixelRatio;
      canvas.height = height * window.devicePixelRatio;
      canvas.style.width = `${width}px`;
      canvas.style.height = `${height}px`;
      ctx.scale(window.devicePixelRatio, window.devicePixelRatio);
      if (orbsRef.current.length === 0) {
        initOrbs(width, height);
      }
    };

    const handleMouse = (e: MouseEvent) => {
      mouseRef.current = {
        x: e.clientX / width,
        y: e.clientY / height,
      };
    };

    resize();
    window.addEventListener("resize", resize, { passive: true });
    window.addEventListener("mousemove", handleMouse, { passive: true });

    const drawGrid = (time: number) => {
      ctx.strokeStyle = "rgba(99, 140, 130, 0.04)";
      ctx.lineWidth = 0.5;

      const spacing = 60;
      const offset = (time * 0.01) % spacing;

      // Horizontal lines
      for (let y = -spacing + offset; y < height + spacing; y += spacing) {
        ctx.beginPath();
        ctx.moveTo(0, y);
        ctx.lineTo(width, y);
        ctx.stroke();
      }

      // Vertical lines
      for (let x = -spacing + offset; x < width + spacing; x += spacing) {
        ctx.beginPath();
        ctx.moveTo(x, 0);
        ctx.lineTo(x, height);
        ctx.stroke();
      }
    };

    const drawOrbs = (time: number) => {
      const mx = mouseRef.current.x;
      const my = mouseRef.current.y;

      orbsRef.current.forEach((orb) => {
        // Organic movement
        orb.x += orb.vx + Math.sin(time * 0.0005 + orb.phase) * 0.3;
        orb.y += orb.vy + Math.cos(time * 0.0004 + orb.phase) * 0.3;

        // Mouse influence
        const dx = mx * width - orb.x;
        const dy = my * height - orb.y;
        const dist = Math.sqrt(dx * dx + dy * dy);
        if (dist < 400) {
          orb.x += dx * 0.001;
          orb.y += dy * 0.001;
        }

        // Wrap around
        if (orb.x < -orb.radius) orb.x = width + orb.radius;
        if (orb.x > width + orb.radius) orb.x = -orb.radius;
        if (orb.y < -orb.radius) orb.y = height + orb.radius;
        if (orb.y > height + orb.radius) orb.y = -orb.radius;

        // Pulsing radius
        const pulseRadius =
          orb.radius + Math.sin(time * 0.001 + orb.phase) * 30;

        // Draw radial gradient orb
        const gradient = ctx.createRadialGradient(
          orb.x,
          orb.y,
          0,
          orb.x,
          orb.y,
          pulseRadius
        );
        gradient.addColorStop(0, `${orb.color} ${orb.opacity})`);
        gradient.addColorStop(0.5, `${orb.color} ${orb.opacity * 0.5})`);
        gradient.addColorStop(1, `${orb.color} 0)`);

        ctx.fillStyle = gradient;
        ctx.beginPath();
        ctx.arc(orb.x, orb.y, pulseRadius, 0, Math.PI * 2);
        ctx.fill();
      });
    };

    const drawParticles = (time: number) => {
      particlesRef.current.forEach((p) => {
        p.life += 1;
        if (p.life > p.maxLife) {
          p.life = 0;
          p.x = Math.random() * width;
          p.y = Math.random() * height;
        }

        p.x += p.vx + Math.sin(time * 0.001 + p.x * 0.01) * 0.1;
        p.y += p.vy + Math.cos(time * 0.001 + p.y * 0.01) * 0.1;

        // Fade in/out
        const lifeRatio = p.life / p.maxLife;
        const fade =
          lifeRatio < 0.1
            ? lifeRatio / 0.1
            : lifeRatio > 0.9
            ? (1 - lifeRatio) / 0.1
            : 1;

        ctx.fillStyle = `rgba(99, 140, 130, ${p.opacity * fade})`;
        ctx.beginPath();
        ctx.arc(p.x, p.y, p.size, 0, Math.PI * 2);
        ctx.fill();
      });

      // Draw connections between close particles
      ctx.strokeStyle = "rgba(99, 140, 130, 0.03)";
      ctx.lineWidth = 0.5;

      for (let i = 0; i < particlesRef.current.length; i++) {
        for (let j = i + 1; j < particlesRef.current.length; j++) {
          const a = particlesRef.current[i];
          const b = particlesRef.current[j];
          const dx = a.x - b.x;
          const dy = a.y - b.y;
          const dist = Math.sqrt(dx * dx + dy * dy);

          if (dist < 120) {
            const alpha = (1 - dist / 120) * 0.06;
            ctx.strokeStyle = `rgba(99, 140, 130, ${alpha})`;
            ctx.beginPath();
            ctx.moveTo(a.x, a.y);
            ctx.lineTo(b.x, b.y);
            ctx.stroke();
          }
        }
      }
    };

    const animate = (timestamp: number) => {
      timeRef.current = timestamp;
      ctx.clearRect(0, 0, width, height);

      drawGrid(timestamp);
      drawOrbs(timestamp);
      drawParticles(timestamp);

      frameRef.current = requestAnimationFrame(animate);
    };

    frameRef.current = requestAnimationFrame(animate);

    return () => {
      cancelAnimationFrame(frameRef.current);
      window.removeEventListener("resize", resize);
      window.removeEventListener("mousemove", handleMouse);
    };
  }, [initOrbs]);

  return (
    <canvas
      ref={canvasRef}
      className="fixed inset-0 -z-10 pointer-events-none"
      style={{ background: "#0D1B17" }}
    />
  );
}

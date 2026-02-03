import * as React from "react";
import { cva, type VariantProps } from "class-variance-authority";

const badgeVariants = cva(
  "inline-flex items-center rounded-full border px-2.5 py-0.5 text-xs font-semibold transition-colors focus:outline-none focus:ring-2 focus:ring-ring focus:ring-offset-2",
  {
    variants: {
      variant: {
        default: "border-transparent bg-white/10 text-white",
        secondary: "border-transparent bg-white/5 text-slate-400",
        destructive: "border-transparent bg-red-500/20 text-red-500",
        success: "border-transparent bg-green-500/20 text-green-500",
        warning: "border-transparent bg-yellow-500/20 text-yellow-500",
        outline: "border-white/20 text-white",
        gradient: "border-transparent bg-spike-gradient text-white",
      },
    },
    defaultVariants: {
      variant: "default",
    },
  }
);

export interface BadgeProps
  extends React.HTMLAttributes<HTMLDivElement>,
    VariantProps<typeof badgeVariants> {}

function Badge({ className, variant, ...props }: BadgeProps) {
  return (
    <div
      className={badgeVariants({ variant })}
      style={{ className }}
      {...props}
    />
  );
}

export { Badge, badgeVariants };

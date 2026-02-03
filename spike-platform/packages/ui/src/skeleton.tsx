import * as React from "react";

function Skeleton({
  className,
  ...props
}: React.HTMLAttributes<HTMLDivElement>) {
  return (
    <div
      className={`animate-pulse rounded-lg bg-white/10 ${className || ""}`}
      {...props}
    />
  );
}

export { Skeleton };

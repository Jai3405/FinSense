import * as React from "react";

interface SectionProps extends React.HTMLAttributes<HTMLElement> {
  title?: string;
  description?: string;
  action?: React.ReactNode;
}

export function Section({
  title,
  description,
  action,
  className,
  children,
  ...props
}: SectionProps) {
  return (
    <section className={`space-y-4 ${className || ""}`} {...props}>
      {(title || description || action) && (
        <div className="flex items-center justify-between">
          <div>
            {title && (
              <h2 className="text-lg font-semibold text-white">{title}</h2>
            )}
            {description && (
              <p className="text-sm text-slate-400 mt-1">{description}</p>
            )}
          </div>
          {action}
        </div>
      )}
      {children}
    </section>
  );
}

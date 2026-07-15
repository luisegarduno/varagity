import * as React from "react"
import { cva, type VariantProps } from "class-variance-authority"

import { cn } from "@/lib/utils"

const badgeVariants = cva(
  "inline-flex w-fit shrink-0 items-center justify-center gap-1 whitespace-nowrap border border-transparent font-medium tabular-nums transition-colors [&>svg]:pointer-events-none [&>svg]:shrink-0",
  {
    variants: {
      variant: {
        default: "bg-muted text-muted-foreground",
        accent:
          "bg-primary/10 text-primary dark:bg-primary/15 dark:text-[oklch(0.78_calc(var(--accent-chroma)*0.7)_var(--accent-hue))]",
        outline: "border-border bg-transparent text-foreground",
        success:
          "bg-emerald-600/10 text-emerald-700 dark:bg-emerald-400/10 dark:text-emerald-400",
        destructive: "bg-destructive/10 text-destructive dark:bg-destructive/15",
        warning:
          "bg-amber-600/10 text-amber-700 dark:bg-amber-400/10 dark:text-amber-400",
      },
      size: {
        sm: "h-5 rounded-sm px-1.5 text-[0.6875rem] [&>svg]:size-3",
        md: "h-6 rounded-md px-2 text-xs [&>svg]:size-3.5",
      },
    },
    defaultVariants: {
      variant: "default",
      size: "sm",
    },
  }
)

function Badge({
  className,
  variant = "default",
  size = "sm",
  ...props
}: React.ComponentProps<"span"> & VariantProps<typeof badgeVariants>) {
  return (
    <span
      data-slot="badge"
      className={cn(badgeVariants({ variant, size, className }))}
      {...props}
    />
  )
}

export { Badge, badgeVariants }

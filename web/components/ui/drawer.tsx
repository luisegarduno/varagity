"use client"

import * as React from "react"
import { Drawer as DrawerPrimitive } from "@base-ui/react/drawer"
import { cva } from "class-variance-authority"

import { cn } from "@/lib/utils"

type DrawerSide = "bottom" | "left"

/** Lets DrawerContent inherit the side chosen on the root. */
const DrawerSideContext = React.createContext<DrawerSide>("bottom")

function Drawer({
  side = "bottom",
  swipeDirection,
  ...props
}: DrawerPrimitive.Root.Props & { side?: DrawerSide }) {
  return (
    <DrawerSideContext.Provider value={side}>
      <DrawerPrimitive.Root
        data-slot="drawer"
        swipeDirection={swipeDirection ?? (side === "bottom" ? "down" : "left")}
        {...props}
      />
    </DrawerSideContext.Provider>
  )
}

function DrawerTrigger({ ...props }: DrawerPrimitive.Trigger.Props) {
  return <DrawerPrimitive.Trigger data-slot="drawer-trigger" {...props} />
}

function DrawerPortal({ ...props }: DrawerPrimitive.Portal.Props) {
  return <DrawerPrimitive.Portal data-slot="drawer-portal" {...props} />
}

function DrawerClose({ ...props }: DrawerPrimitive.Close.Props) {
  return <DrawerPrimitive.Close data-slot="drawer-close" {...props} />
}

function DrawerBackdrop({
  className,
  ...props
}: DrawerPrimitive.Backdrop.Props) {
  return (
    <DrawerPrimitive.Backdrop
      data-slot="drawer-backdrop"
      className={cn(
        "fixed inset-0 z-50 min-h-dvh bg-black/25 opacity-[calc(1-var(--drawer-swipe-progress))] transition-opacity duration-[450ms] ease-[cubic-bezier(0.32,0.72,0,1)] supports-backdrop-filter:backdrop-blur-xs data-swiping:duration-0 data-starting-style:opacity-0 data-ending-style:opacity-0 data-ending-style:duration-[calc(var(--drawer-swipe-strength)*400ms)] motion-reduce:transition-none dark:bg-black/45",
        className
      )}
      {...props}
    />
  )
}

const drawerViewportVariants = cva("fixed inset-0 z-50 flex", {
  variants: {
    side: {
      bottom: "items-end justify-center",
      left: "items-stretch justify-start",
    },
  },
  defaultVariants: { side: "bottom" },
})

function DrawerViewport({
  className,
  side = "bottom",
  ...props
}: DrawerPrimitive.Viewport.Props & { side?: DrawerSide }) {
  return (
    <DrawerPrimitive.Viewport
      data-slot="drawer-viewport"
      className={cn(drawerViewportVariants({ side }), className)}
      {...props}
    />
  )
}

/* The 3rem "bleed" hangs off-screen so rubber-band over-drag never reveals
   a gap; the enter/exit transform compensates for it (Base UI drives
   `--drawer-swipe-movement-*` while swiping and scales the exit duration
   by `--drawer-swipe-strength`). */
const drawerPopupVariants = cva(
  "relative z-50 flex flex-col bg-background text-foreground outline-none will-change-transform transition-transform duration-[450ms] ease-[cubic-bezier(0.32,0.72,0,1)] overflow-y-auto overscroll-contain touch-auto data-swiping:select-none data-swiping:duration-0 data-ending-style:duration-[calc(var(--drawer-swipe-strength)*400ms)] motion-reduce:transition-none",
  {
    variants: {
      side: {
        bottom:
          "-mb-12 w-full max-h-[calc(85dvh+3rem)] rounded-t-2xl border-t border-border pb-[calc(3rem+env(safe-area-inset-bottom,0px))] shadow-[0_-8px_32px_-12px_rgb(0_0_0/0.18)] [transform:translateY(var(--drawer-swipe-movement-y))] data-starting-style:[transform:translateY(calc(100%-3rem+2px))] data-ending-style:[transform:translateY(calc(100%-3rem+2px))]",
        left: "-ml-12 h-full w-[calc(19rem+3rem)] max-w-[100vw] border-r border-border pl-12 shadow-[8px_0_32px_-12px_rgb(0_0_0/0.18)] [transform:translateX(var(--drawer-swipe-movement-x))] data-starting-style:[transform:translateX(calc(-100%+3rem-2px))] data-ending-style:[transform:translateX(calc(-100%+3rem-2px))]",
      },
    },
    defaultVariants: { side: "bottom" },
  }
)

function DrawerPopup({
  className,
  side = "bottom",
  ...props
}: DrawerPrimitive.Popup.Props & { side?: DrawerSide }) {
  return (
    <DrawerPrimitive.Popup
      data-slot="drawer-popup"
      className={cn(drawerPopupVariants({ side }), className)}
      {...props}
    />
  )
}

function DrawerHandle({ className, ...props }: React.ComponentProps<"div">) {
  return (
    <div
      data-slot="drawer-handle"
      aria-hidden
      className={cn(
        "mx-auto mt-2.5 mb-3 h-1 w-9 shrink-0 rounded-full bg-muted-foreground/25",
        className
      )}
      {...props}
    />
  )
}

function DrawerContent({
  className,
  contentClassName,
  children,
  side: sideProp,
  showHandle = true,
  ...props
}: DrawerPrimitive.Popup.Props & {
  side?: DrawerSide
  showHandle?: boolean
  /** Extra classes for the inner `Drawer.Content` (e.g. `p-0` to go full-bleed). */
  contentClassName?: string
}) {
  const contextSide = React.useContext(DrawerSideContext)
  const side = sideProp ?? contextSide
  return (
    <DrawerPortal>
      <DrawerBackdrop />
      <DrawerViewport side={side}>
        <DrawerPopup side={side} className={className} {...props}>
          {side === "bottom" && showHandle && <DrawerHandle />}
          <DrawerPrimitive.Content
            data-slot="drawer-content"
            className={cn(
              "flex w-full min-w-0 flex-1 flex-col gap-4 px-5 pb-5",
              (side === "left" || !showHandle) && "pt-5",
              contentClassName
            )}
          >
            {children}
          </DrawerPrimitive.Content>
        </DrawerPopup>
      </DrawerViewport>
    </DrawerPortal>
  )
}

function DrawerHeader({ className, ...props }: React.ComponentProps<"div">) {
  return (
    <div
      data-slot="drawer-header"
      className={cn("flex flex-col gap-1.5", className)}
      {...props}
    />
  )
}

function DrawerFooter({ className, ...props }: React.ComponentProps<"div">) {
  return (
    <div
      data-slot="drawer-footer"
      className={cn(
        "mt-auto flex flex-col-reverse gap-2 sm:flex-row sm:justify-end",
        className
      )}
      {...props}
    />
  )
}

function DrawerTitle({ className, ...props }: DrawerPrimitive.Title.Props) {
  return (
    <DrawerPrimitive.Title
      data-slot="drawer-title"
      className={cn("font-heading text-lg leading-snug font-normal", className)}
      {...props}
    />
  )
}

function DrawerDescription({
  className,
  ...props
}: DrawerPrimitive.Description.Props) {
  return (
    <DrawerPrimitive.Description
      data-slot="drawer-description"
      className={cn("text-sm text-muted-foreground", className)}
      {...props}
    />
  )
}

export {
  Drawer,
  DrawerBackdrop,
  DrawerClose,
  DrawerContent,
  DrawerDescription,
  DrawerFooter,
  DrawerHandle,
  DrawerHeader,
  DrawerPopup,
  DrawerPortal,
  DrawerTitle,
  DrawerTrigger,
  DrawerViewport,
}

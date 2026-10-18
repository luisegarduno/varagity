/**
 * Entry of the docs-site map bundle (ADR-018): mounts the very component the
 * Next app serves at `/map`, outside the app shell. The only difference from
 * the app is where the favicons live — beside this page, because GitHub
 * Pages serves the site under a repository subpath (see `icon-base.ts`).
 * Theme follows the host docs page through the inline script in index.html;
 * nothing here fetches anything.
 */

import { StrictMode } from "react";
import { createRoot } from "react-dom/client";

import { CodebaseMapView } from "@/components/map/CodebaseMapView";
import { MapIconBaseContext } from "@/components/map/icon-base";

import "./map.css";

const root = document.getElementById("root");
if (root === null) throw new Error("docs-map: #root is missing from index.html");

createRoot(root).render(
  <StrictMode>
    <MapIconBaseContext value="map-icons">
      <CodebaseMapView />
    </MapIconBaseContext>
  </StrictMode>,
);

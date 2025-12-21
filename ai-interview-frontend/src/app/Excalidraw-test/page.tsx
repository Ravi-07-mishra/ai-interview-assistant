"use client";
import React, { useState, useEffect } from "react";
import dynamic from "next/dynamic";
import "@excalidraw/excalidraw/index.css";

const Excalidraw = dynamic(
  () => import("@excalidraw/excalidraw").then((mod) => mod.Excalidraw),
  { ssr: false }
);

export default function ExcalidrawDebugger() {
  const [mounted, setMounted] = useState(false);

  useEffect(() => {
    setMounted(true);
    
    // GLOBAL CLICK DEBUGGER
    // This will tell us if an invisible modal/div is stealing your clicks
    const debugClick = (e: MouseEvent) => {
      console.log("🖱️ CLICKED ON:", e.target);
    };
    window.addEventListener("click", debugClick);
    return () => window.removeEventListener("click", debugClick);
  }, []);

  if (!mounted) return <div className="p-10">Loading...</div>;

  return (
    <div style={{ height: "100vh", width: "100vw", overflow: "hidden" }}>
      <Excalidraw
        // 1. Force View Mode OFF explicitly
        viewModeEnabled={false} 
        // 2. Force Zen Mode OFF
        zenModeEnabled={false}
        // 3. Force Grid Mode ON (Visual confirmation it's rendering)
        gridModeEnabled={true} 
      />
    </div>
  );
}
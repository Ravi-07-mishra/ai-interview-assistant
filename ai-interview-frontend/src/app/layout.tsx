// src/app/layout.tsx
import "./globals.css";
import React from "react";
import { AuthProvider } from "./context/AuthContext";
export const metadata = {
  title: "AI Interview",
  description: "Upload resume and practice interviews",
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en">
      <head>
        {/* ensure responsive viewport */}
        <meta name="viewport" content="width=device-width, initial-scale=1" />
      </head>
      <body>
        <AuthProvider>{children}</AuthProvider>
      </body>
    </html>
  );
}

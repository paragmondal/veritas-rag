import type { Metadata, Viewport } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "Veritas — Enterprise Hybrid RAG",
  description: "Enterprise hybrid search and retrieval-augmented generation over corporate filings.",
};

export const viewport: Viewport = {
  themeColor: [
    { media: "(prefers-color-scheme: light)", color: "#F5F5F7" },
    { media: "(prefers-color-scheme: dark)", color: "#000000" },
  ],
  width: "device-width",
  initialScale: 1,
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en" className="h-full">
      <body className="h-full bg-apple-bgLight dark:bg-apple-bgDark text-apple-textLight dark:text-apple-textDark transition-colors duration-200">
        {children}
      </body>
    </html>
  );
}

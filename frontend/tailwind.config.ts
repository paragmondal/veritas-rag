import type { Config } from "tailwindcss";

const config: Config = {
  darkMode: "class",
  content: [
    "./app/**/*.{js,ts,jsx,tsx,mdx}",
    "./components/**/*.{js,ts,jsx,tsx,mdx}",
  ],
  theme: {
    extend: {
      fontFamily: {
        sans: [
          "-apple-system",
          "BlinkMacSystemFont",
          '"SF Pro Text"',
          '"SF Pro Display"',
          '"Helvetica Neue"',
          "sans-serif",
        ],
      },
      colors: {
        apple: {
          bgLight: "#F5F5F7",
          cardLight: "#FFFFFF",
          bgDark: "#000000",
          cardDark: "#1C1C1E",
          cardDarkHover: "#2C2C2E",
          accentLight: "#0071E3",
          accentDark: "#0A84FF",
          borderLight: "rgba(0, 0, 0, 0.08)",
          borderDark: "rgba(255, 255, 255, 0.12)",
          textLight: "#1D1D1F",
          textDark: "#F5F5F7",
          subtextLight: "#86868B",
          subtextDark: "#98989D",
          userBubble: "#0071E3",
          assistantBubbleLight: "#FFFFFF",
          assistantBubbleDark: "#1C1C1E",
        },
      },
      boxShadow: {
        apple: "0 1px 3px rgba(0, 0, 0, 0.08)",
        appleModal: "0 12px 36px rgba(0, 0, 0, 0.18)",
      },
      borderRadius: {
        card: "16px",
        button: "10px",
        tag: "9999px",
      },
    },
  },
  plugins: [],
};

export default config;

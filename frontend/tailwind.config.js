/** @type {import('tailwindcss').Config} */
export default {
  content: ["./index.html", "./src/**/*.{js,ts,jsx,tsx}"],
  theme: {
    extend: {
      colors: {
        medical: {
          50: "#ecfeff",
          100: "#cffafe",
          200: "#a5f3fc",
          500: "#06b6d4",
          600: "#0891b2",
          700: "#0e7490",
          800: "#155e75",
          900: "#164e63",
        },
      },
      fontFamily: {
        sans: [
          "-apple-system",
          "BlinkMacSystemFont",
          '"PingFang SC"',
          '"Hiragino Sans GB"',
          '"Microsoft YaHei"',
          '"SF Pro Text"',
          "Inter",
          "sans-serif",
        ],
        mono: [
          '"JetBrains Mono"',
          '"SF Mono"',
          "Menlo",
          "Monaco",
          "Consolas",
          "monospace",
        ],
      },
      boxShadow: {
        glass:
          "0 12px 36px -12px rgba(15, 23, 42, 0.08), 0 4px 12px -4px rgba(14, 116, 144, 0.05), inset 0 1px 0 rgba(255, 255, 255, 0.9)",
        "glass-sm":
          "0 6px 20px -8px rgba(15, 23, 42, 0.07), inset 0 1px 0 rgba(255, 255, 255, 0.85)",
        "glass-danger":
          "0 10px 28px -10px rgba(190, 18, 60, 0.14), inset 0 1px 0 rgba(255, 255, 255, 0.78)",
      },
    },
  },
  plugins: [],
};

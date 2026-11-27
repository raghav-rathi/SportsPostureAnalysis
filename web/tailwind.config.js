export default {
  content: ["./index.html", "./src/**/*.{js,ts,jsx,tsx}"],
  theme: {
    extend: {
      colors: {
        paper: "#fafafa",
        pure: "#ffffff",
        subtle: "#f4f4f5",
        hairline: "#ebebeb",
        ash: "#c9c9c9",
        smoke: "#a8a8a8",
        graphite: "#8f8f8f",
        slatev: "#7d7d7d",
        stone: "#666666",
        charcoal: "#4d4d4d",
        obsidian: "#171717",
        carbon: "#000000",
        terminal: "#297a3a",
      },
      fontFamily: {
        sans: ['"Geist"', '"Inter"', "-apple-system", "BlinkMacSystemFont", '"Segoe UI"', "sans-serif"],
        mono: ['"Geist Mono"', "ui-monospace", "SFMono-Regular", "Menlo", "Monaco", "Consolas", "monospace"],
      },
      borderRadius: {
        v: "6px",
      },
    },
  },
  plugins: [],
};

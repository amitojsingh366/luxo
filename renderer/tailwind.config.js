/** @type {import('tailwindcss').Config} */
export default {
  content: ["./index.html", "./src/**/*.{ts,html}"],
  theme: {
    extend: {
      colors: {
        ink: "#06080d",
        ember: "#f3b96b",
      },
      fontFamily: {
        display: ["Avenir Next", "Avenir", "Inter", "sans-serif"],
      },
    },
  },
  plugins: [],
};

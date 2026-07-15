import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "Helicopter Scoreboard",
  description: "RWKV evaluation scoreboard"
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="zh-CN">
      <body>{children}</body>
    </html>
  );
}

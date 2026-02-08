import { auth } from "@clerk/nextjs/server";
import { redirect } from "next/navigation";
import { Sidebar } from "@/components/layout/sidebar";
import { TopBar } from "@/components/layout/topbar";

export default async function DashboardLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  const { userId } = await auth();

  if (!userId) {
    redirect("/sign-in");
  }

  return (
    <div className="min-h-screen" style={{ backgroundColor: "#F5FFFC" }}>
      <Sidebar />
      <div className="lg:pl-64">
        <TopBar />
        <main className="py-5 px-6">{children}</main>
      </div>
    </div>
  );
}

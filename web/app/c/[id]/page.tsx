import { Conversation } from "@/components/chat/Conversation";

/** One conversation. `params` is a Promise in Next 16 — await it. */
export default async function ConversationPage({
  params,
}: {
  params: Promise<{ id: string }>;
}) {
  const { id } = await params;
  // Key by id: switching conversations remounts with fresh chat state.
  return <Conversation key={id} conversationId={id} />;
}

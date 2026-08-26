"""Document manager component for Streamlit."""

import streamlit as st
import requests


def render_document_manager(backend_url: str):
    """Renders the document upload, URL crawler, and index management UI."""
    st.subheader("📁 Document Knowledge Base")

    tab_upload, tab_url = st.tabs(["📤 Upload Files", "🌐 Ingest Web URL"])

    with tab_upload:
        uploaded_files = st.file_uploader(
            "Upload Documents (PDF, DOCX, Markdown, HTML, JSON, CSV, TSV, TXT)",
            type=["pdf", "docx", "md", "markdown", "txt", "csv", "tsv", "json", "html", "htm", "log"],
            accept_multiple_files=True,
            help="Upload documents to be parsed, chunked, and indexed in FAISS + BM25."
        )

        col1, col2 = st.columns([3, 1])
        with col1:
            if st.button("🚀 Process & Ingest Files", type="primary", use_container_width=True, key="btn_ingest_files"):
                if not uploaded_files:
                    st.warning("Please select at least one document to upload.")
                else:
                    progress_bar = st.progress(0)
                    for idx, file in enumerate(uploaded_files):
                        with st.spinner(f"Ingesting & indexing '{file.name}'..."):
                            files = {"file": (file.name, file.getvalue(), file.type)}
                            try:
                                res = requests.post(f"{backend_url}/api/v1/documents/upload", files=files)
                                if res.status_code == 201:
                                    data = res.json()
                                    st.success(f"✅ {file.name} indexed ({data['chunks_created']} chunks in {data['time_taken_ms']}ms)")
                                else:
                                    st.error(f"❌ Failed to ingest {file.name}: {res.text}")
                            except Exception as e:
                                st.error(f"Connection error: {e}")
                        progress_bar.progress((idx + 1) / len(uploaded_files))
                    st.rerun()

        with col2:
            if st.button("🗑️ Clear All Docs", use_container_width=True, key="btn_clear_docs"):
                try:
                    res = requests.delete(f"{backend_url}/api/v1/documents")
                    if res.status_code == 200:
                        st.success("All indexes cleared.")
                        st.rerun()
                    else:
                        st.error(f"Error: {res.text}")
                except Exception as e:
                    st.error(f"Connection error: {e}")

    with tab_url:
        st.markdown("Fetch articles, web documentation, or public reports directly into the RAG vector store:")
        url_input = st.text_input("Web Page URL", placeholder="https://en.wikipedia.org/wiki/Retrieval-augmented_generation")

        if st.button("🌐 Scrape & Ingest URL", type="primary", use_container_width=True, key="btn_ingest_url"):
            if not url_input.strip():
                st.warning("Please enter a valid URL.")
            else:
                with st.spinner(f"Fetching and indexing '{url_input}'..."):
                    try:
                        res = requests.post(
                            f"{backend_url}/api/v1/documents/url",
                            json={"url": url_input.strip()},
                            timeout=30
                        )
                        if res.status_code == 201:
                            data = res.json()
                            st.success(f"✅ Scraped & Indexed '{url_input}' ({data['chunks_created']} chunks in {data['time_taken_ms']}ms)")
                            st.rerun()
                        else:
                            st.error(f"❌ Failed to ingest URL: {res.text}")
                    except Exception as e:
                        st.error(f"Connection error: {e}")

    # Indexed Documents Table
    st.markdown("---")
    st.markdown("### 📑 Currently Indexed Documents")
    try:
        res = requests.get(f"{backend_url}/api/v1/documents")
        if res.status_code == 200:
            doc_data = res.json()
            docs = doc_data.get("documents", [])
            total_chunks = doc_data.get("total_chunks", 0)

            m1, m2 = st.columns(2)
            m1.metric("Indexed Documents", len(docs))
            m2.metric("Total Vector Chunks", total_chunks)

            if not docs:
                st.info("No documents indexed yet. Upload files or ingest a URL above to get started.")
            else:
                for doc in docs:
                    icon = "🌐" if doc.get("file_type") == "url" else "📄"
                    with st.expander(f"{icon} **{doc['filename']}** ({doc['total_chunks']} chunks)"):
                        st.write(f"**Doc ID:** `{doc['doc_id']}`")
                        st.write(f"**Format:** `{doc['file_type'].upper()}` | **Size:** {round(doc['file_size_bytes']/1024, 1)} KB")
                        if doc.get("total_pages"):
                            st.write(f"**Sections / Pages:** {doc['total_pages']}")
                        if doc.get("custom_metadata") and doc["custom_metadata"].get("source_url"):
                            st.write(f"**Source URL:** [{doc['custom_metadata']['source_url']}]({doc['custom_metadata']['source_url']})")
                        st.write(f"**Indexed At:** {doc['created_at']}")

                        if st.button("Delete Document", key=f"del_{doc['doc_id']}"):
                            del_res = requests.delete(f"{backend_url}/api/v1/documents/{doc['doc_id']}")
                            if del_res.status_code == 200:
                                st.success(f"Deleted {doc['filename']}")
                                st.rerun()
                            else:
                                st.error(f"Delete failed: {del_res.text}")
        else:
            st.error("Failed to fetch documents from backend.")
    except Exception as e:
        st.warning(f"Backend API offline or unreachable at {backend_url}. Details: {e}")

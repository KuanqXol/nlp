"""
Module: graph_visualization.py
Chức năng: Trực quan hóa Knowledge Graph bằng networkx + pyvis.

Output:
  - File HTML tương tác (pyvis)
  - Plot tĩnh (matplotlib) nếu pyvis không có

Tính năng:
  - Màu sắc theo loại entity (PER=xanh, LOC=xanh lá, ORG=đỏ, MISC=vàng)
  - Kích thước node theo PageRank score
  - Label trên edge là relation
  - Lọc hiển thị theo top-K entity quan trọng nhất
"""

import os
from typing import Dict, List, Optional

# ── Try import visualization libraries ──────────────────────────────────────

try:
    import networkx as nx

    _NX = True
except ImportError:
    _NX = False

try:
    from pyvis.network import Network

    _PYVIS = True
except ImportError:
    _PYVIS = False

try:
    import matplotlib

    matplotlib.use("Agg")  # non-interactive backend
    import matplotlib.pyplot as plt

    _MPL = True
except ImportError:
    _MPL = False


# ── Color mapping ────────────────────────────────────────────────────────────

ENTITY_COLORS = {
    "PER": "#4A90D9",  # Xanh dương → Người
    "LOC": "#27AE60",  # Xanh lá   → Địa điểm
    "ORG": "#E74C3C",  # Đỏ        → Tổ chức
    "MISC": "#F39C12",  # Cam       → Khác
    "UNK": "#95A5A6",  # Xám       → Không xác định
}

ENTITY_SHAPES = {
    "PER": "dot",
    "LOC": "triangle",
    "ORG": "square",
    "MISC": "diamond",
}


# ── Subgraph extractor ───────────────────────────────────────────────────────


def extract_subgraph(kg, top_k: int = 50, min_degree: int = 1):
    """
    Trích xuất subgraph gồm top-k entity quan trọng nhất.
    Giảm complexity khi KG quá lớn.
    """
    if not _NX:
        return None

    # Sắp xếp theo PageRank (hoặc degree nếu chưa có PageRank)
    pagerank = kg._pagerank
    if pagerank:
        top_nodes = sorted(pagerank.keys(), key=lambda x: -pagerank[x])[:top_k]
    else:
        degrees = dict(kg.graph.degree())
        top_nodes = sorted(degrees.keys(), key=lambda x: -degrees[x])[:top_k]

    # Lấy induced subgraph
    subgraph = kg.graph.subgraph(top_nodes).copy()

    # Lọc isolated nodes (degree = 0)
    if min_degree > 0:
        to_remove = [n for n, d in subgraph.degree() if d < min_degree]
        subgraph.remove_nodes_from(to_remove)

    return subgraph


# ── Pyvis Interactive HTML ───────────────────────────────────────────────────


class PyvisVisualizer:
    """
    Tạo Knowledge Graph tương tác dạng HTML dùng pyvis.
    Mở bằng trình duyệt web để xem và tương tác.
    """

    def __init__(
        self,
        height: str = "750px",
        width: str = "100%",
        bgcolor: str = "#1a1a2e",
        font_color: str = "#eaeaea",
    ):
        self.height = height
        self.width = width
        self.bgcolor = bgcolor
        self.font_color = font_color

    def visualize(
        self,
        kg,
        output_path: str = "knowledge_graph.html",
        top_k: int = 60,
        title: str = "Vietnamese News Knowledge Graph",
    ) -> Optional[str]:
        """
        Tạo file HTML interactive từ KG.

        Args:
            kg: KnowledgeGraph instance
            output_path: Đường dẫn file HTML output
            top_k: Hiển thị top-k entity quan trọng nhất
            title: Tiêu đề của visualize

        Returns:
            Path đến file HTML, hoặc None nếu lỗi
        """
        if not _PYVIS:
            print("[Visualization] Pyvis chưa cài. pip install pyvis")
            return None
        if not _NX:
            print("[Visualization] Networkx chưa cài. pip install networkx")
            return None

        subgraph = extract_subgraph(kg, top_k=top_k)
        if subgraph is None or subgraph.number_of_nodes() == 0:
            print("[Visualization] Subgraph trống, không có gì để visualize.")
            return None

        print(
            f"[Visualization] Đang render {subgraph.number_of_nodes()} nodes, "
            f"{subgraph.number_of_edges()} edges..."
        )

        # Khởi tạo pyvis network
        net = Network(
            height=self.height,
            width=self.width,
            bgcolor=self.bgcolor,
            font_color=self.font_color,
            directed=True,
            notebook=False,
        )

        # Cấu hình physics
        net.set_options(
            """
        var options = {
          "physics": {
            "forceAtlas2Based": {
              "gravitationalConstant": -80,
              "springLength": 120,
              "springConstant": 0.05
            },
            "solver": "forceAtlas2Based",
            "stabilization": {"iterations": 150}
          },
          "interaction": {
            "hover": true,
            "tooltipDelay": 100
          }
        }
        """
        )

        # Thêm nodes
        pagerank = kg._pagerank
        max_size, min_size = 40, 10

        for node, data in subgraph.nodes(data=True):
            etype = data.get("type", "UNK")
            color = ENTITY_COLORS.get(etype, ENTITY_COLORS["UNK"])
            shape = ENTITY_SHAPES.get(etype, "dot")
            pr = pagerank.get(node, 0.01)
            size = min_size + (max_size - min_size) * min(pr * 10, 1)

            freq = data.get("frequency", 0)
            doc_ids = data.get("doc_ids", [])
            tooltip = (
                f"Entity: {node}\n"
                f"Type: {etype}\n"
                f"PageRank: {pr:.4f}\n"
                f"Frequency: {freq}\n"
                f"Bài báo: {len(doc_ids)}"
            )

            net.add_node(
                node,
                label=node,
                color=color,
                size=int(size),
                shape=shape,
                title=tooltip,
            )

        # Thêm edges
        for u, v, data in subgraph.edges(data=True):
            relations = data.get("relations", [data.get("relation", "")])
            label = relations[0] if relations else ""
            weight = data.get("weight", 1)
            width = max(1, min(int(weight), 5))

            # Bỏ co_occurrence để tránh rối
            if label == "co_occurrence":
                continue

            net.add_edge(
                u,
                v,
                title=label,
                label=label if len(label) < 20 else "",
                width=width,
                arrows="to",
                color={"color": "#7f8c8d", "highlight": "#f39c12"},
            )

        # Thêm legend node (text)
        legend_text = (
            "🔵 PER: Người\n" "🟢 LOC: Địa điểm\n" "🔴 ORG: Tổ chức\n" "🟡 MISC: Khác"
        )

        # Export HTML
        net.show(output_path, notebook=False)
        print(f"[Visualization] Đã lưu Knowledge Graph → {output_path}")
        return output_path


# ── Matplotlib static plot (fallback) ───────────────────────────────────────


class MatplotlibVisualizer:
    """
    Visualize KG tĩnh bằng matplotlib (fallback khi không có pyvis).
    """

    def visualize(
        self,
        kg,
        output_path: str = "knowledge_graph.png",
        top_k: int = 30,
        figsize: tuple = (16, 12),
    ) -> Optional[str]:
        """Tạo PNG của KG."""
        if not _MPL or not _NX:
            print("[Visualization] Cần matplotlib và networkx.")
            return None

        subgraph = extract_subgraph(kg, top_k=top_k)
        if subgraph is None or subgraph.number_of_nodes() == 0:
            return None

        fig, ax = plt.subplots(figsize=figsize)
        ax.set_facecolor("#1a1a2e")
        fig.patch.set_facecolor("#1a1a2e")

        # Layout
        pos = nx.spring_layout(subgraph, k=2.0, seed=42)

        # Node colors và sizes
        node_colors = []
        node_sizes = []
        pagerank = kg._pagerank

        for node in subgraph.nodes():
            etype = subgraph.nodes[node].get("type", "UNK")
            node_colors.append(ENTITY_COLORS.get(etype, ENTITY_COLORS["UNK"]))
            pr = pagerank.get(node, 0.01)
            node_sizes.append(200 + 2000 * min(pr * 5, 1))

        # Draw
        nx.draw_networkx_nodes(
            subgraph,
            pos,
            ax=ax,
            node_color=node_colors,
            node_size=node_sizes,
            alpha=0.9,
        )
        nx.draw_networkx_labels(
            subgraph,
            pos,
            ax=ax,
            font_size=8,
            font_color="white",
        )

        # Edge labels (chỉ vẽ non-cooccurrence)
        edge_labels = {}
        edges_to_draw = []
        for u, v, data in subgraph.edges(data=True):
            rel = data.get("relation", "")
            if rel != "co_occurrence":
                edges_to_draw.append((u, v))
                edge_labels[(u, v)] = rel

        nx.draw_networkx_edges(
            subgraph,
            pos,
            ax=ax,
            edgelist=edges_to_draw,
            edge_color="#7f8c8d",
            arrows=True,
            arrowsize=10,
            alpha=0.6,
        )
        nx.draw_networkx_edge_labels(
            subgraph,
            pos,
            ax=ax,
            edge_labels=edge_labels,
            font_size=6,
            font_color="#f39c12",
        )

        ax.set_title(
            "Vietnamese News Knowledge Graph",
            color="white",
            fontsize=14,
            pad=15,
        )
        ax.axis("off")

        # Legend
        from matplotlib.patches import Patch

        legend_elements = [
            Patch(facecolor=ENTITY_COLORS["PER"], label="PER (Người)"),
            Patch(facecolor=ENTITY_COLORS["LOC"], label="LOC (Địa điểm)"),
            Patch(facecolor=ENTITY_COLORS["ORG"], label="ORG (Tổ chức)"),
            Patch(facecolor=ENTITY_COLORS["MISC"], label="MISC (Khác)"),
        ]
        ax.legend(
            handles=legend_elements,
            loc="upper left",
            facecolor="#2c3e50",
            edgecolor="white",
            labelcolor="white",
        )

        plt.tight_layout()
        plt.savefig(
            output_path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor()
        )
        plt.close()
        print(f"[Visualization] Đã lưu ảnh KG → {output_path}")
        return output_path


# ── Unified Visualizer ───────────────────────────────────────────────────────


class KnowledgeGraphVisualizer:
    """
    Bọc cả PyvisVisualizer và MatplotlibVisualizer.
    Tự động chọn backend phù hợp.
    """

    def visualize(
        self,
        kg,
        output_path: str = "knowledge_graph",
        top_k: int = 50,
        interactive: bool = True,
    ) -> Optional[str]:
        """
        Visualize KG và lưu file.

        Args:
            kg: KnowledgeGraph instance
            output_path: Prefix đường dẫn (không cần extension)
            top_k: Số entity hiển thị
            interactive: Nếu True → dùng pyvis HTML; ngược lại → matplotlib PNG

        Returns:
            Đường dẫn file output
        """
        if interactive and _PYVIS:
            viz = PyvisVisualizer()
            return viz.visualize(kg, output_path + ".html", top_k=top_k)
        elif _MPL:
            viz = MatplotlibVisualizer()
            return viz.visualize(kg, output_path + ".png", top_k=top_k)
        else:
            print(
                "[Visualization] Không có thư viện visualization nào (pyvis/matplotlib)."
            )
            return None


# ── Demo standalone ──────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).parents[2]))
    from src.graph.knowledge_graph import KnowledgeGraph
    from src.graph.ranking import GraphRanker

    kg = KnowledgeGraph()
    docs = [
        {
            "id": "d1",
            "linked_entities": [
                {"canonical": "Putin", "type": "PER"},
                {"canonical": "Nga", "type": "LOC"},
                {"canonical": "Ukraine", "type": "LOC"},
                {"canonical": "Zelensky", "type": "PER"},
                {"canonical": "NATO", "type": "ORG"},
                {"canonical": "WHO", "type": "ORG"},
                {"canonical": "COVID-19", "type": "MISC"},
                {"canonical": "Việt Nam", "type": "LOC"},
                {"canonical": "Hà Nội", "type": "LOC"},
                {"canonical": "VinAI", "type": "ORG"},
                {"canonical": "Phạm Minh Chính", "type": "PER"},
            ],
            "triples": [
                {"subject": "Putin", "relation": "lãnh đạo", "object": "Nga"},
                {"subject": "Nga", "relation": "tấn công", "object": "Ukraine"},
                {"subject": "Zelensky", "relation": "lãnh đạo", "object": "Ukraine"},
                {"subject": "NATO", "relation": "hỗ trợ", "object": "Ukraine"},
                {"subject": "WHO", "relation": "cảnh báo", "object": "COVID-19"},
                {"subject": "VinAI", "relation": "đặt tại", "object": "Hà Nội"},
                {"subject": "Hà Nội", "relation": "thuộc", "object": "Việt Nam"},
            ],
        },
    ]
    kg.build_from_documents(docs)

    ranker = GraphRanker()
    ranker.compute_pagerank(kg)

    viz = KnowledgeGraphVisualizer()
    out = viz.visualize(kg, output_path="/tmp/kg_demo", top_k=30)
    if out:
        print(f"Đã tạo visualization: {out}")

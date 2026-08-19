window.ANSWER_VIZ_BUNDLE = {
  "queries": [
    {
      "query_id": "1",
      "query_type": "relational",
      "title": "Vic's testimony — witness conflict",
      "summary": "Party/witness relationships from the hypo, with a qualitative contrast of Vic's live testimony vs the retrieved hearsay excerpt.",
      "primary_view": {
        "type": "network",
        "items": [],
        "layout": "hierarchical",
        "network": {
          "nodes": [
            {
              "id": "issue",
              "label": "Vic testimony?",
              "entity_type": "doctrine",
              "description": "Should the judge admit Vic after Wes testified unfavorably?",
              "evidence_ids": [
                "e1"
              ],
              "layer": 0
            },
            {
              "id": "paul",
              "label": "Paul",
              "entity_type": "plaintiff",
              "description": "Plaintiff; calls Wes, then offers Vic.",
              "evidence_ids": [
                "e1"
              ],
              "layer": 1
            },
            {
              "id": "wes",
              "label": "Wes",
              "entity_type": "actor",
              "description": "Testifies Dan did NOT run the light.",
              "evidence_ids": [
                "e1"
              ],
              "layer": 2
            },
            {
              "id": "vic",
              "label": "Vic",
              "entity_type": "actor",
              "description": "Offered to say Dan DID run the light.",
              "evidence_ids": [
                "e2"
              ],
              "layer": 2
            },
            {
              "id": "dan",
              "label": "Dan",
              "entity_type": "defendant",
              "description": "Driver alleged to have run the light.",
              "evidence_ids": [
                "e1"
              ],
              "layer": 3
            }
          ],
          "edges": [
            {
              "source": "paul",
              "target": "wes",
              "relation": "calls",
              "description": "Paul calls Wes as eyewitness.",
              "evidence_ids": [
                "e1"
              ]
            },
            {
              "source": "wes",
              "target": "dan",
              "relation": "denies red light",
              "description": "Wes says Dan did not run the light.",
              "evidence_ids": [
                "e1"
              ]
            },
            {
              "source": "paul",
              "target": "vic",
              "relation": "offers",
              "description": "Paul offers Vic to contradict Wes.",
              "evidence_ids": [
                "e2"
              ]
            },
            {
              "source": "vic",
              "target": "dan",
              "relation": "asserts red light",
              "description": "Vic would say Dan ran the light.",
              "evidence_ids": [
                "e2"
              ]
            }
          ]
        }
      },
      "secondary_views": [],
      "legend": [
        {
          "key": "party",
          "label": "Parties / witnesses",
          "color_token": "teal"
        },
        {
          "key": "doctrine",
          "label": "Retrieved doctrine",
          "color_token": "blue"
        }
      ],
      "annotations": [
        {
          "id": "a1",
          "text": "Key story beat: Wes and Vic collide on the same fact.",
          "target_item_id": "t2",
          "evidence_ids": [
            "e1"
          ]
        }
      ],
      "evidence_links": [
        {
          "evidence_id": "e1",
          "document_id": "passages/e451f42aede23c6cd711ed87b0a18dd138c1dee6a0d868525c94d13ab3aa7c83.txt",
          "citation": "passages/e451f42aede23c6cd711ed87b0a18dd138c1dee6a0d868525c94d13ab3aa7c83.txt",
          "passage": "apturing the driver of one car stating that the other driver ran a red light would be hearsay when offered to prove that the other driver ran a red light.",
          "source_url": null
        },
        {
          "evidence_id": "e2",
          "document_id": "passages/4349fc11baa7ce5c0f5f48b0dec7244d604d974755825af1934aba50444b2470.txt",
          "citation": "passages/4349fc11baa7ce5c0f5f48b0dec7244d604d974755825af1934aba50444b2470.txt",
          "passage": "Ms. Primo testified that she witnessed the accident at issue. She testified that she was the first car in line in the left lane on Severn Avenue, facing Airline, waiting on the red light to turn green. She testified that she was sitting at the red light listening to music and looking straight ahead on Severn Avenue when she saw Mr. Windmann's car \"out of the left peripheral vision.\" According to Ms. Primo, Mr. Murray was riding his bike on the street about two or three car lengths away from the ",
          "source_url": null
        },
        {
          "evidence_id": "e3",
          "document_id": "passages/5247b5c1e7da6fcf16b80be2ae1eb0b8222cd68534fddee29865f6d8f12fefb8.txt",
          "citation": "passages/5247b5c1e7da6fcf16b80be2ae1eb0b8222cd68534fddee29865f6d8f12fefb8.txt",
          "passage": "In this case, the Plaintiff claims that Dan's Car World was told by its Zurich insurance agent, Scott Beresford, that Salvatore was covered under the insurance issued to Dan's Car World. While Tiffany Moine, Dan's Car World's Rule 30(b)(7) deponent, testified that she did not know what Mr. Beresford specifically told Dan Cava, she also testified that she was made aware by Dan Cava of a conversation between himself and Zurich's agent on adding Salvatore to Dan's Car World's policy. And after that",
          "source_url": null
        },
        {
          "evidence_id": "e4",
          "document_id": "passages/3ed089eb7d86a3b564458c302b2cd58c351eec389f64aba8deb5ce5b9072cc88.txt",
          "citation": "passages/3ed089eb7d86a3b564458c302b2cd58c351eec389f64aba8deb5ce5b9072cc88.txt",
          "passage": "Reduced to its essence, Ms. Moine testified that Mr. Cava told her that he asked Petitioners' agent to add Salvatore Cava to the policy. After Ms. Moine's deposition the Petitioners filed a motion to compel Dan's Car World to produce a Rule 30(b)(7) witness who could testify about what its agent said to Mr. Cava regarding adding Salvatore Cava to the policy. Dan's Car World opposed the motion. In doing so, it argued, \"to the extent [Dan's Car World's] representative was unaware of information so",
          "source_url": null
        },
        {
          "evidence_id": "e5",
          "document_id": "passages/5247b5c1e7da6fcf16b80be2ae1eb0b8222cd68534fddee29865f6d8f12fefb8.txt",
          "citation": "passages/5247b5c1e7da6fcf16b80be2ae1eb0b8222cd68534fddee29865f6d8f12fefb8.txt",
          "passage": "alvatore to Dan's Car World's policy. And after that conversation, Mr. Cava was \"under the assumption that\" Salvatore had been added to the policy. Like Ms. Moine, Dan Cava testified that he had a conversation with Mr. Beresford in which he wanted to add Salvatore to Dan's Car World's policy as a named insured and Mr. Beresford advised him that his son was covered. The positions taken by Dan's Car World and Dan Cava are not inconsistent, as both Dan's Car World and Dan Cava testified regarding a",
          "source_url": null
        }
      ],
      "limitations": [
        "The provided evidence does not contain sufficient legal rules or precedents to determine the admissibility of Vic's testimony.",
        "The distinction between hearsay (Excerpt 1) and direct witness testimony is crucial but not fully addressed by the available documents.",
        "Story visuals summarize the hypo + model answer; they are not a judicial ruling."
      ],
      "fallback_text": "The available legal documents do not provide enough information to determine how the trial judge should rule on Vic's testimony. The excerpts discuss hearsay and factual scenarios, but not the specific rules of evidence applicable to direct witness testimony in this context.",
      "_source": {
        "query": "Paul, the Plaintiff in a personal injury action, called Wes as a witness to testify that Dan's car, in which Paul had been riding. ran a red light. Wes, however, testified that Dan's car did not run the light.\nPaul then called Vic to testify that Dan's car did run the light. The trial judge should rule that Vic's testimony is",
        "generated_answer": "The provided legal document excerpts do not contain sufficient information to determine how the trial judge should rule on Vic's testimony. Excerpt 1 discusses hearsay in the context of \"the driver of one car stating that the other driver ran a red light,\" which is distinct from a witness (Vic) directly testifying to what they observed. The other excerpts describe factual scenarios or deposition testimony and do not provide general rules of evidence applicable to Vic's direct testimony.",
        "model_id": "gemini-2.5-flash",
        "backend": "gemini"
      },
      "_story": {
        "takeaway": "Direct eyewitness offer vs out-of-court hearsay excerpt — retrieved law does not clearly decide Vic.",
        "beats": []
      },
      "_story_timeline": {
        "items": []
      },
      "limited_evidence": true,
      "established_facts": [
        "Paul called Wes; Wes testified Dan's car did not run the light. (query)",
        "Paul then offered Vic to testify that Dan's car did run the light. (query)",
        "Retrieved excerpt discusses hearsay for an out-of-court driver statement about a red light. (e1)"
      ],
      "not_established": [
        "How the trial judge should rule on Vic's testimony.",
        "A retrieved rule of evidence that directly governs Vic's live in-court testimony."
      ],
      "_audit": {
        "backed_items": 24,
        "unsupported_items": 0,
        "fabricated_numeric": 0,
        "limited_evidence": true,
        "primary": "network",
        "diagram_kind": null
      },
      "additional_details": {
        "type": "matrix",
        "items": [
          {
            "id": "m1",
            "row": "Vic live",
            "column": "In-court",
            "value": "Yes",
            "description": "Offered as live witness.",
            "evidence_ids": [
              "e1"
            ]
          },
          {
            "id": "m2",
            "row": "Vic live",
            "column": "Covered by e1 hearsay?",
            "value": "No",
            "description": "e1 is about out-of-court statements.",
            "evidence_ids": [
              "e1"
            ]
          },
          {
            "id": "m3",
            "row": "e1 driver statement",
            "column": "In-court",
            "value": "No",
            "description": "Out-of-court assertion.",
            "evidence_ids": [
              "e1"
            ]
          },
          {
            "id": "m4",
            "row": "e1 driver statement",
            "column": "Covered by e1 hearsay?",
            "value": "Yes",
            "description": "Classic hearsay hypo in retrieval.",
            "evidence_ids": [
              "e1"
            ]
          },
          {
            "id": "m5",
            "row": "Rule deciding Vic",
            "column": "In-court",
            "value": "Missing",
            "description": "Not in retrieved set.",
            "evidence_ids": [
              "e5"
            ]
          },
          {
            "id": "m6",
            "row": "Rule deciding Vic",
            "column": "Covered by e1 hearsay?",
            "value": "N/A",
            "description": "Wrong analogy.",
            "evidence_ids": [
              "e5"
            ]
          }
        ]
      }
    },
    {
      "query_id": "2",
      "query_type": "comparison",
      "title": "Impeaching Vic — prior conduct after a broad denial",
      "summary": "Qualitative comparison of what the hypo asks versus what retrieved passages establish about admitting Yank.",
      "primary_view": {
        "type": "matrix",
        "x_label": "Question",
        "y_label": "Strand",
        "items": [
          {
            "id": "m1",
            "row": "Accident-night intoxication",
            "column": "In hypo",
            "value": "Yes",
            "symbol": "✓",
            "description": "Asked on cross of Vic.",
            "evidence_ids": [
              "e1"
            ]
          },
          {
            "id": "m2",
            "row": "Accident-night intoxication",
            "column": "Yank proves it",
            "value": "No",
            "symbol": "✗",
            "description": "Yank concerns a prior New Year's Eve.",
            "evidence_ids": [
              "e2"
            ]
          },
          {
            "id": "m3",
            "row": "Prior New Year's Eve (Yank)",
            "column": "In hypo",
            "value": "Yes",
            "symbol": "✓",
            "description": "Extrinsic offer of proof.",
            "evidence_ids": [
              "e2"
            ]
          },
          {
            "id": "m4",
            "row": "Prior New Year's Eve (Yank)",
            "column": "Yank proves it",
            "value": "Offered",
            "symbol": "○",
            "description": "What counsel seeks to prove.",
            "evidence_ids": [
              "e2"
            ]
          },
          {
            "id": "m5",
            "row": "Prior-acts impeachment rule",
            "column": "In hypo",
            "value": "Asked",
            "symbol": "?",
            "description": "The ruling question.",
            "evidence_ids": [
              "e5"
            ]
          },
          {
            "id": "m6",
            "row": "Prior-acts impeachment rule",
            "column": "Yank proves it",
            "value": "Missing",
            "symbol": "—",
            "description": "Not found in retrieved passages.",
            "evidence_ids": [
              "e5"
            ]
          }
        ]
      },
      "secondary_views": [],
      "legend": [
        {
          "key": "compare",
          "label": "On-point vs retrieved support",
          "color_token": "amber"
        }
      ],
      "annotations": [],
      "evidence_links": [
        {
          "evidence_id": "e1",
          "document_id": "passages/831f0c32daf282018c45220a706920e511fcd0ffd4bd00683716bd7a23a5f9eb.txt",
          "citation": "passages/831f0c32daf282018c45220a706920e511fcd0ffd4bd00683716bd7a23a5f9eb.txt",
          "passage": "ted he had stabbed the victim, but he did not think the wound was bad enough to kill him. On cross-examination at trial, defense counsel tried to show the victim had a reputation for fighting, and the defendant had acted in self defense. Another witness testified the victim was drunk that night and was known for fighting. Gregory also admitted they had been drinking and testified the victim had been in fights in the past. A different witness, however, testified the victim was angry but not drunk",
          "source_url": null
        },
        {
          "evidence_id": "e2",
          "document_id": "passages/cac408a05b7cccee656e493c04480dfc0641c7e7c28e67cfd61723b8c417df10.txt",
          "citation": "passages/cac408a05b7cccee656e493c04480dfc0641c7e7c28e67cfd61723b8c417df10.txt",
          "passage": "The main issue in this case was whether Burnette pulled his car to the side or left it in the middle of the road. This determination hinged upon the credibility of the witnesses. Thus, whether Burnette's intoxication after the accident affected his ability to recall where he parked his vehicle or if he was actually intoxicated at the time he parked it was highly relevant and should have been presented for the jury to consider. Under the trial court's broad ruling, Barrett was prohibited from eve",
          "source_url": null
        },
        {
          "evidence_id": "e3",
          "document_id": "passages/3ed089eb7d86a3b564458c302b2cd58c351eec389f64aba8deb5ce5b9072cc88.txt",
          "citation": "passages/3ed089eb7d86a3b564458c302b2cd58c351eec389f64aba8deb5ce5b9072cc88.txt",
          "passage": "Reduced to its essence, Ms. Moine testified that Mr. Cava told her that he asked Petitioners' agent to add Salvatore Cava to the policy. After Ms. Moine's deposition the Petitioners filed a motion to compel Dan's Car World to produce a Rule 30(b)(7) witness who could testify about what its agent said to Mr. Cava regarding adding Salvatore Cava to the policy. Dan's Car World opposed the motion. In doing so, it argued, \"to the extent [Dan's Car World's] representative was unaware of information so",
          "source_url": null
        },
        {
          "evidence_id": "e4",
          "document_id": "passages/5247b5c1e7da6fcf16b80be2ae1eb0b8222cd68534fddee29865f6d8f12fefb8.txt",
          "citation": "passages/5247b5c1e7da6fcf16b80be2ae1eb0b8222cd68534fddee29865f6d8f12fefb8.txt",
          "passage": "alvatore to Dan's Car World's policy. And after that conversation, Mr. Cava was \"under the assumption that\" Salvatore had been added to the policy. Like Ms. Moine, Dan Cava testified that he had a conversation with Mr. Beresford in which he wanted to add Salvatore to Dan's Car World's policy as a named insured and Mr. Beresford advised him that his son was covered. The positions taken by Dan's Car World and Dan Cava are not inconsistent, as both Dan's Car World and Dan Cava testified regarding a",
          "source_url": null
        },
        {
          "evidence_id": "e5",
          "document_id": "passages/c789266709d90acd8f30270ce9309ddbe319b904873645e7b54586e6d989bb1a.txt",
          "citation": "passages/c789266709d90acd8f30270ce9309ddbe319b904873645e7b54586e6d989bb1a.txt",
          "passage": " sufficient because there was testimony that the defendant was weaving on the road, the defendant's eyes were bloodshot, and the defendant said that he had been drinking);",
          "source_url": null
        }
      ],
      "limitations": [
        "The provided legal document excerpts do not contain information regarding the admissibility of specific instances of prior conduct to impeach a witness's general statement about their past behavior, particularly when that conduct is unrelated to the event in question.",
        "The trial judge's ruling on Yank's testimony cannot be determined from the provided context due to the absence of relevant legal principles or case law in the retrieved evidence."
      ],
      "fallback_text": "The legal documents provided do not contain sufficient information to determine the trial judge's ruling on the admissibility of Yank's testimony regarding Vic's prior conduct, as the excerpts do not address the specific legal principles for impeaching a witness's general statement with unrelated prior acts.",
      "_source": {
        "query": "Paul, the Plaintiff in a personal injury action, called Wes as a witness to testify that Dan's car, in which Paul had been riding. ran a red light. Wes, however, testified that Dan's car did not run the light.\nOn cross-examination of Vic, Dan's attorney asked if Vic was drunk at the time he witnessed the accident. and Vic responded, \"No I have never in my life been drunk.\" Dan's attorney then sought to prove by Yank that Vic was drunk on New Year's Eve two years before the accident. The trialjudge should rule that Yank's testimony is",
        "generated_answer": "The provided legal document excerpts do not contain information regarding the admissibility of specific instances of prior conduct to impeach a witness's general statement about their past behavior, particularly when that conduct is unrelated to the event in question. Therefore, the trial judge's ruling on Yank's testimony cannot be determined from the provided context.",
        "model_id": "gemini-2.5-flash",
        "backend": "gemini"
      },
      "_story": {
        "takeaway": "The hypo poses an impeachment question the retrieved passages do not answer.",
        "beats": []
      },
      "_story_timeline": {
        "items": []
      },
      "limited_evidence": true,
      "established_facts": [
        "On cross, Vic denied ever having been drunk. (query)",
        "Dan's attorney sought to prove by Yank that Vic was drunk on a prior New Year's Eve. (query)",
        "Retrieved passages discuss intoxication/credibility in other cases, not a clear prior-acts impeachment rule for this hypo. (e1, e2)"
      ],
      "not_established": [
        "Whether Yank's testimony is admissible.",
        "A governing doctrine in the retrieved set for impeaching a lifetime sobriety claim with a specific unrelated prior act."
      ],
      "_audit": {
        "backed_items": 23,
        "unsupported_items": 0,
        "fabricated_numeric": 0,
        "limited_evidence": true,
        "primary": "matrix",
        "diagram_kind": null
      },
      "additional_details": {
        "type": "network",
        "layout": "hierarchical",
        "network": {
          "nodes": [
            {
              "id": "atty",
              "label": "Dan's atty",
              "entity_type": "actor",
              "description": "Seeks impeachment via Yank.",
              "evidence_ids": [
                "e1"
              ],
              "layer": 0
            },
            {
              "id": "vic",
              "label": "Vic",
              "entity_type": "actor",
              "description": "Claims never been drunk.",
              "evidence_ids": [
                "e1"
              ],
              "layer": 1
            },
            {
              "id": "yank",
              "label": "Yank",
              "entity_type": "third_party",
              "description": "Would testify to prior drunkenness.",
              "evidence_ids": [
                "e2"
              ],
              "layer": 1
            },
            {
              "id": "rule",
              "label": "Prior-acts rule",
              "entity_type": "doctrine",
              "description": "Not established in retrieval.",
              "evidence_ids": [
                "e5"
              ],
              "layer": 2
            }
          ],
          "edges": [
            {
              "source": "atty",
              "target": "vic",
              "relation": "cross-examines",
              "description": "",
              "evidence_ids": [
                "e1"
              ]
            },
            {
              "source": "atty",
              "target": "yank",
              "relation": "offers",
              "description": "",
              "evidence_ids": [
                "e2"
              ]
            },
            {
              "source": "yank",
              "target": "vic",
              "relation": "prior act",
              "description": "",
              "evidence_ids": [
                "e2"
              ]
            },
            {
              "source": "rule",
              "target": "yank",
              "relation": "missing",
              "description": "",
              "evidence_ids": [
                "e5"
              ]
            }
          ]
        }
      }
    },
    {
      "query_id": "3",
      "query_type": "relational",
      "title": "Zemo — Vic's reputation for veracity",
      "summary": "Who asks whom about reputation, and what the retrieved record does and does not establish.",
      "primary_view": {
        "type": "network",
        "layout": "hierarchical",
        "items": [],
        "network": {
          "nodes": [
            {
              "id": "issue",
              "label": "Reputation ask OK?",
              "entity_type": "doctrine",
              "description": "Is Zemo's question about Vic's veracity reputation admissible?",
              "evidence_ids": [
                "e1"
              ],
              "layer": 0
            },
            {
              "id": "dan",
              "label": "Dan",
              "entity_type": "defendant",
              "description": "Calls Zemo.",
              "evidence_ids": [
                "e1"
              ],
              "layer": 1
            },
            {
              "id": "zemo",
              "label": "Zemo",
              "entity_type": "actor",
              "description": "Asked about Vic's community reputation for veracity.",
              "evidence_ids": [
                "e1"
              ],
              "layer": 2
            },
            {
              "id": "vic",
              "label": "Vic",
              "entity_type": "actor",
              "description": "Target of the reputation question.",
              "evidence_ids": [
                "e1"
              ],
              "layer": 3
            },
            {
              "id": "rule",
              "label": "Veracity rule",
              "entity_type": "doctrine",
              "description": "Not established in retrieved passages.",
              "evidence_ids": [
                "e5"
              ],
              "layer": 1
            }
          ],
          "edges": [
            {
              "source": "dan",
              "target": "zemo",
              "relation": "calls",
              "description": "Dan calls Zemo.",
              "evidence_ids": [
                "e1"
              ]
            },
            {
              "source": "zemo",
              "target": "vic",
              "relation": "reputation?",
              "description": "Asked about community veracity reputation.",
              "evidence_ids": [
                "e1"
              ]
            },
            {
              "source": "dan",
              "target": "issue",
              "relation": "poses",
              "description": "Admissibility is the open question.",
              "evidence_ids": [
                "e1"
              ]
            },
            {
              "source": "rule",
              "target": "issue",
              "relation": "missing",
              "description": "Governing doctrine absent from retrieval.",
              "evidence_ids": [
                "e5"
              ]
            }
          ]
        }
      },
      "secondary_views": [],
      "legend": [],
      "annotations": [],
      "evidence_links": [
        {
          "evidence_id": "e1",
          "document_id": "passages/e451f42aede23c6cd711ed87b0a18dd138c1dee6a0d868525c94d13ab3aa7c83.txt",
          "citation": "passages/e451f42aede23c6cd711ed87b0a18dd138c1dee6a0d868525c94d13ab3aa7c83.txt",
          "passage": "apturing the driver of one car stating that the other driver ran a red light would be hearsay when offered to prove that the other driver ran a red light.",
          "source_url": null
        },
        {
          "evidence_id": "e2",
          "document_id": "passages/2166fbe348000df2f904b3c489cf250b78dcce21abf87c92ef8090b88bdf5e55.txt",
          "citation": "passages/2166fbe348000df2f904b3c489cf250b78dcce21abf87c92ef8090b88bdf5e55.txt",
          "passage": "This case presented conflicting testimony between the parties and the trial court assessed the credibility of the witnesses in determining who ran the red light. In assessing credibility, the trial court stated that Adams's testimony seemed credible, but questioned the account that Adams had time to warn his brother of the accident. The trial court noted that Adams had a stake in the outcome, his testimony was self-serving, and he is a convicted felon. The trial court found Price's testimony to ",
          "source_url": null
        },
        {
          "evidence_id": "e3",
          "document_id": "passages/5247b5c1e7da6fcf16b80be2ae1eb0b8222cd68534fddee29865f6d8f12fefb8.txt",
          "citation": "passages/5247b5c1e7da6fcf16b80be2ae1eb0b8222cd68534fddee29865f6d8f12fefb8.txt",
          "passage": "alvatore to Dan's Car World's policy. And after that conversation, Mr. Cava was \"under the assumption that\" Salvatore had been added to the policy. Like Ms. Moine, Dan Cava testified that he had a conversation with Mr. Beresford in which he wanted to add Salvatore to Dan's Car World's policy as a named insured and Mr. Beresford advised him that his son was covered. The positions taken by Dan's Car World and Dan Cava are not inconsistent, as both Dan's Car World and Dan Cava testified regarding a",
          "source_url": null
        },
        {
          "evidence_id": "e4",
          "document_id": "passages/5247b5c1e7da6fcf16b80be2ae1eb0b8222cd68534fddee29865f6d8f12fefb8.txt",
          "citation": "passages/5247b5c1e7da6fcf16b80be2ae1eb0b8222cd68534fddee29865f6d8f12fefb8.txt",
          "passage": "In this case, the Plaintiff claims that Dan's Car World was told by its Zurich insurance agent, Scott Beresford, that Salvatore was covered under the insurance issued to Dan's Car World. While Tiffany Moine, Dan's Car World's Rule 30(b)(7) deponent, testified that she did not know what Mr. Beresford specifically told Dan Cava, she also testified that she was made aware by Dan Cava of a conversation between himself and Zurich's agent on adding Salvatore to Dan's Car World's policy. And after that",
          "source_url": null
        },
        {
          "evidence_id": "e5",
          "document_id": "passages/31de716a0f053ec2309112a3deed68c2798fb828fbd783a771ad04f702ebcf0e.txt",
          "citation": "passages/31de716a0f053ec2309112a3deed68c2798fb828fbd783a771ad04f702ebcf0e.txt",
          "passage": "id not know. In response to Petitioners' motion to compel Dan's Car World to produce a Rule 30(b)(7) witness who could answer the question, Dan's Car World stated that the information was not available. Several years later Mr. Cava testified at a deposition, called by the Respondent, that Petitioners' agent informed him that Salvatore Cava was already covered under the garage provision and umbrella provision of the policy. Nowhere in the record does it show that at the time of Ms. Moine's deposi",
          "source_url": null
        }
      ],
      "limitations": [
        "The answer is strictly limited to the information contained within the provided legal document excerpts. No external legal knowledge or analysis was applied.",
        "The retrieved passages did not contain specific legal rules or precedents regarding the admissibility of questions about a witness's reputation for veracity in the community.",
        "Network shows the evidentiary move, not an authority for admitting reputation evidence."
      ],
      "fallback_text": "The provided legal documents do not contain information regarding the admissibility of a question about a witness's reputation for veracity in the community. Therefore, the trial judge's ruling on this question cannot be determined from the context.",
      "_source": {
        "query": "Paul, the Plaintiff in a personal injury action, called Wes as a witness to testify that Dan's car, in which Paul had been riding. ran a red light. Wes, however, testified that Dan's car did not run the light.\nDan called Zemo as a witness and asked him if he knew Vic's reputation for veracity in the community where Vic resided. The trialjudge should rule that this question is",
        "generated_answer": "The provided legal document excerpts do not contain information regarding the admissibility of a question about a witness's reputation for veracity in the community. Therefore, the trial judge's ruling on this question cannot be determined from the context.",
        "model_id": "gemini-2.5-flash",
        "backend": "gemini"
      },
      "_story": {
        "takeaway": "Reputation-for-veracity is posed by the hypo but not resolved by retrieval.",
        "beats": []
      },
      "_story_timeline": {
        "items": []
      },
      "limited_evidence": true,
      "established_facts": [
        "Dan called Zemo and asked about Vic's reputation for veracity in the community. (query)",
        "Vic's credibility is material because Wes and Vic conflict on the red-light fact. (query)"
      ],
      "not_established": [
        "Whether the reputation question is admissible.",
        "A retrieved rule governing reputation-for-veracity evidence in this setting."
      ],
      "_audit": {
        "backed_items": 19,
        "unsupported_items": 0,
        "fabricated_numeric": 0,
        "limited_evidence": true,
        "primary": "network",
        "diagram_kind": null
      },
      "additional_details": {
        "type": "matrix",
        "items": [
          {
            "id": "m1",
            "row": "Vic credibility at issue",
            "column": "Established",
            "value": "Yes",
            "symbol": "✓",
            "description": "Wes vs Vic conflict.",
            "evidence_ids": [
              "e1"
            ]
          },
          {
            "id": "m2",
            "row": "Reputation question asked",
            "column": "Established",
            "value": "Yes",
            "symbol": "✓",
            "description": "Zemo question in hypo.",
            "evidence_ids": [
              "e1"
            ]
          },
          {
            "id": "m3",
            "row": "Governing doctrine retrieved",
            "column": "Established",
            "value": "No",
            "symbol": "✗",
            "description": "Not in passages.",
            "evidence_ids": [
              "e5"
            ]
          },
          {
            "id": "m4",
            "row": "Confident ruling",
            "column": "Established",
            "value": "No",
            "symbol": "✗",
            "description": "Matches generated answer.",
            "evidence_ids": [
              "e5"
            ]
          }
        ]
      }
    },
    {
      "query_id": "4",
      "query_type": "temporal",
      "title": "Jim's burglary — intent at entry",
      "summary": "Sequence of events from the hypo, linked to the generated defense theory and retrieved intent-at-entry language.",
      "primary_view": {
        "type": "timeline",
        "x_label": "Sequence (from hypo; no calendar dates in evidence)",
        "items": [
          {
            "id": "t1",
            "date": null,
            "date_label": "Stage 1 · Planning",
            "date_certainty": "sequence_only",
            "event": "Planned hold-up; bought toy gun",
            "description": "Stated in the query narrative.",
            "category": "preparation",
            "evidence_ids": [
              "e1"
            ]
          },
          {
            "id": "t2",
            "date": null,
            "date_label": "Stage 2 · Entry",
            "date_certainty": "sequence_only",
            "event": "Entered store near closing",
            "description": "Critical burglary moment in the hypo.",
            "category": "entry",
            "evidence_ids": [
              "e2"
            ]
          },
          {
            "id": "t3",
            "date": null,
            "date_label": "Stage 3 · Retreat",
            "date_certainty": "sequence_only",
            "event": "Frightened; moved toward the door",
            "description": "From the query; used in the defense theory.",
            "category": "retreat",
            "evidence_ids": [
              "e2"
            ]
          },
          {
            "id": "t4",
            "date": null,
            "date_label": "Stage 4 · Defense",
            "date_certainty": "sequence_only",
            "event": "Argue no felony intent at entry",
            "description": "Generated answer + retrieved intent-at-entry language (e2).",
            "category": "legal",
            "evidence_ids": [
              "e2"
            ]
          }
        ]
      },
      "secondary_views": [],
      "legend": [],
      "annotations": [],
      "evidence_links": [
        {
          "evidence_id": "e1",
          "document_id": "passages/7567ea4c221c6faa2cf168c47348dde9bd23ac58c4571cda72324c86fc00e829.txt",
          "citation": "passages/7567ea4c221c6faa2cf168c47348dde9bd23ac58c4571cda72324c86fc00e829.txt",
          "passage": "He entered through the front door; pointed a black handgun at Kenny Butler; and watched Butler place cash from the register, cigarettes, and Patrón tequila (two pint-sized bottles in green boxes) into a bag. The videos also show the robber placing his hands and gun on the service counter, right in front of Butler, and ripping apart the store's phone. They next display the robber moving his gun toward his waist or sweatshirt pocket as he exited the same door he entered. One video shows him walkin",
          "source_url": null
        },
        {
          "evidence_id": "e2",
          "document_id": "passages/041ac5eddc891d9c4f84f99a951daef5c31c04560d32df3a8f8473855dd47393.txt",
          "citation": "passages/041ac5eddc891d9c4f84f99a951daef5c31c04560d32df3a8f8473855dd47393.txt",
          "passage": "ot follow that Soto 's reasoning bears on burglary charges-which are based on the defendant's intent at the moment of entering a store, regardless of whether violence actually occurred.",
          "source_url": null
        },
        {
          "evidence_id": "e3",
          "document_id": "passages/0b684607f3703b741702b200bc6a0671cac531001e55ab580e474f6781e422e2.txt",
          "citation": "passages/0b684607f3703b741702b200bc6a0671cac531001e55ab580e474f6781e422e2.txt",
          "passage": "The state charged Martin with committing this offense by possessing the handgun during the burglary of the alterations shop, See OCGA § 16-11-106 (b) (2) (offense is committed, among other ways, when person has \"on or within arm's reach of his ... person a firearm ... during the commission of ... [t]he unlawful entry into a building\"). Martin argues that, because there were no eyewitnesses to the burglary of the alterations shop, there was no evidence that he possessed the gun when he entered th",
          "source_url": null
        },
        {
          "evidence_id": "e4",
          "document_id": "passages/a3eccf2ce1001a2ac678fa5247d0aa5dd273e3bd93b4d2306f8677254a8a13a1.txt",
          "citation": "passages/a3eccf2ce1001a2ac678fa5247d0aa5dd273e3bd93b4d2306f8677254a8a13a1.txt",
          "passage": "Finally, Petitioner explained how he wound up shooting the victim. He said he took the money and was backing out of the shop without looking at the door. He heard someone coming to the door and turned around-at that point, the victim grabbed a gun. Petitioner turned back, apparently saw the gun, and fired a shot. This shot, Petitioner claimed, was the one that hit the counter. Petitioner backed out of the shop and ran into a person trying to enter. By this time, the victim had fired his own shot",
          "source_url": null
        },
        {
          "evidence_id": "e5",
          "document_id": "passages/95a684dda31260841eb1eef8f27b1af13d8c9d8478f66ac8e635cd691ad0e82b.txt",
          "citation": "passages/95a684dda31260841eb1eef8f27b1af13d8c9d8478f66ac8e635cd691ad0e82b.txt",
          "passage": "¶14. Osbourne said his heart was still pounding from the incident on I-55. He got out of his van and grabbed his gun. He explained that he always carries a gun because he often has to travel to Memphis, Tennessee. Osbourne said he started walking toward the door of the store, when McCarver quickly walked up to him and put his chest against his. Osbourne said that he did not hit McCarver but instead that he pushed McCarver away from him. He claimed that McCarver reached for something in his right",
          "source_url": null
        }
      ],
      "limitations": [
        "Timeline steps are grounded in the query narrative and generated answer.",
        "Retrieved cases may describe similar robberies; they are not a holding on this exact hypo."
      ],
      "fallback_text": "Jim's best defense on a charge of burglary would be that he lacked the intent to commit a felony or to steal at the moment he entered the store.\n\nBurglary charges are \"based on the defendant's intent at the moment of entering a store\" (Court: unknown, Citation: passages/041ac5eddc891d9c4f84f99a951daef5c31c04560d32df3a8f8473855dd47393.txt). Although Jim had planned the hold-up and brought a toy gun, his immediate fright and retreat upon entering the store could be argued as evidence that he did not possess the requisite intent to commit a felony or to steal *at the precise moment he entered*.",
      "_source": {
        "query": "nan\nJim watched a liquor store furtively for some time, planning to hold it up. He bought a realistic-looking toy gun for the job. One night, just before the store's closing time, he drove to the store, opened the front door and entered. He reached in his pocket for the toy gun, but then became frightened and began to move back toward the front door. However, the shopkeeper had seen the butt of the gun. Fearing a hold up, the shopkeeper produced a gun from under the counter, pointed it at Jim, and yelled, \"Stop!\" Jim ran to the door and the toy gun fell from his pocket. The shopkeeper fired. The shot missed Jim, but struck and killed a passerby outside the store. A statute in the jurisdiction defines burglary as \"breaking and entering any building or structure with the intent to commit a felony or to steal therein.\" On a charge of burglary, Jim's best defense would be that",
        "generated_answer": "Jim's best defense on a charge of burglary would be that he lacked the intent to commit a felony or to steal at the moment he entered the store.\n\nBurglary charges are \"based on the defendant's intent at the moment of entering a store\" (Court: unknown, Citation: passages/041ac5eddc891d9c4f84f99a951daef5c31c04560d32df3a8f8473855dd47393.txt). Although Jim had planned the hold-up and brought a toy gun, his immediate fright and retreat upon entering the store could be argued as evidence that he did not possess the requisite intent to commit a felony or to steal *at the precise moment he entered*.",
        "model_id": "gemini-2.5-flash",
        "backend": "gemini"
      },
      "_story": {
        "takeaway": "Timing of intent at the threshold is the defense hinge.",
        "beats": []
      },
      "_story_timeline": {
        "items": []
      },
      "limited_evidence": false,
      "established_facts": [
        "Jim planned a hold-up, bought a toy gun, entered near closing, then became frightened and moved back toward the door. (query)",
        "Retrieved passage: burglary charges are based on intent at the moment of entering a store. (e2)",
        "Generated answer: best defense is lack of felony/theft intent at entry."
      ],
      "not_established": [
        "That a court has already ruled this exact hypo in Jim's favor."
      ],
      "_audit": {
        "backed_items": 19,
        "unsupported_items": 0,
        "fabricated_numeric": 0,
        "limited_evidence": false,
        "primary": "timeline",
        "diagram_kind": null
      },
      "additional_details": {
        "type": "evidence_view",
        "items": [
          {
            "id": "claim-1",
            "title": "Generated defense",
            "text": "Jim's best defense on a charge of burglary would be that he lacked the intent to commit a felony or to steal at the moment he entered the store.\n\nBurglary charges are \"based on the defendant's intent at the moment of entering a store\" (Court: unknown, Citation: passages/041ac5eddc891d9c4f84f99a951daef5c31c04560d32df3a8f8473855dd47393.txt). Although Jim had planned the hold-up and brought a toy gun, his immediate fright and retreat upon entering the store could be argued as evidence that he did n",
            "role": "claim",
            "evidence_ids": [
              "e1",
              "e2"
            ]
          },
          {
            "id": "ev-e2",
            "title": "Retrieved intent-at-entry language",
            "text": "Burglary charges are based on intent at the moment of entering a store (e2).",
            "role": "evidence",
            "evidence_ids": [
              "e2"
            ]
          }
        ]
      }
    },
    {
      "query_id": "5",
      "query_type": "geographic",
      "title": "Res ipsa — chain of control over the tuna",
      "summary": "Chain-of-control diagram across handling stages, plus party graph and res ipsa element checklist. No verified GPS coordinates.",
      "primary_view": {
        "type": "map",
        "x_label": "Chain of control",
        "items": [
          {
            "id": "loc1",
            "location_name": "Canco shipment",
            "region": "Manufacturer",
            "latitude": null,
            "longitude": null,
            "category": "manufacturer",
            "description": "Undamaged cans from same shipment were fit (query).",
            "evidence_ids": [
              "e1"
            ]
          },
          {
            "id": "loc2",
            "location_name": "Supermart basement",
            "region": "Third-party intervention",
            "latitude": null,
            "longitude": null,
            "category": "store",
            "description": "Workmen dent cans during flood repair (query).",
            "evidence_ids": [
              "e1"
            ]
          },
          {
            "id": "loc3",
            "location_name": "Sales floor",
            "region": "Retail",
            "latitude": null,
            "longitude": null,
            "category": "retail",
            "description": "Dented cans sold; Dotty purchases (query).",
            "evidence_ids": [
              "e2"
            ]
          },
          {
            "id": "loc4",
            "location_name": "Dotty",
            "region": "End user",
            "latitude": null,
            "longitude": null,
            "category": "plaintiff",
            "description": "Illness; negligence claim vs Canco (query).",
            "evidence_ids": [
              "e1"
            ]
          }
        ],
        "flow_edges": [
          {
            "source": "loc1",
            "target": "loc2",
            "relation": "leaves Canco control",
            "evidence_ids": [
              "e1"
            ]
          },
          {
            "source": "loc2",
            "target": "loc3",
            "relation": "dented goods sold",
            "evidence_ids": [
              "e2"
            ]
          },
          {
            "source": "loc3",
            "target": "loc4",
            "relation": "purchase / injury",
            "evidence_ids": [
              "e1"
            ]
          }
        ]
      },
      "secondary_views": [],
      "legend": [],
      "annotations": [],
      "evidence_links": [
        {
          "evidence_id": "e1",
          "document_id": "passages/bd91df6820930784d537c29b07af23bb5fa602c7d5bb1a1bcaf2ed5a420e4c13.txt",
          "citation": "passages/bd91df6820930784d537c29b07af23bb5fa602c7d5bb1a1bcaf2ed5a420e4c13.txt",
          "passage": "Res ipsa loquitur is a doctrine addressed to those situations where the facts or circumstances accompanying an injury by their very nature raise a presumption of negligence on the part of [the] defendant.” Bowlin v. Duke Univ., 108 N.C.App. 145, 149, 423 S.E.2d 320, 322 (1992). The doctrine of res ipsa loquitur, “ ‘in its distinctive sense, permits negligence to be inferred from the physical cause of an [injury], without the aid of circumstances pointing to the responsible human cause. Where thi",
          "source_url": null
        },
        {
          "evidence_id": "e2",
          "document_id": "passages/172956519cfadbe23bf79af81adde97036dbce90739133a6db22de95afe733e8.txt",
          "citation": "passages/172956519cfadbe23bf79af81adde97036dbce90739133a6db22de95afe733e8.txt",
          "passage": "hose cans. One can had a puncture hole in the side. Another was sealed at the top. That can was dented on the side, which had allowed the beer to drain out, rendering it unlikely that someone had consumed that beer. Regardless of the condition of the cans, it remains clear that no evidence at trial established that Clemons consumed the beers. More importantly, even if he did drink them, nothing in the record supports the notion that doing so rendered Clemons so intoxicated that he lost control o",
          "source_url": null
        },
        {
          "evidence_id": "e3",
          "document_id": "passages/31bb11f398693184ceb2a3b888a582f331ada6d297a4c184c6ad5319e3de6aeb.txt",
          "citation": "passages/31bb11f398693184ceb2a3b888a582f331ada6d297a4c184c6ad5319e3de6aeb.txt",
          "passage": "The res ipsa loquitur doctrine “enables a jury presented only with circumstantial evidence to infer negligence simply from the fact that an event happened.” St. Paul Fire & Marine Ins. Co. v. City of New York, 907 F.2d 299, 302 (2d Cir.1990).  The criteria for applying res ipsa loquitur include  “(1) the event must be of a kind which ordinarily does not occur in the absence of someone's negligence; (2) it must be caused by an agency or instrumentality within the exclusive control of the defendan",
          "source_url": null
        },
        {
          "evidence_id": "e4",
          "document_id": "passages/230900df234728a7cb5fcaad1d7c7f500b374fbf06593313468f50c99c79d244.txt",
          "citation": "passages/230900df234728a7cb5fcaad1d7c7f500b374fbf06593313468f50c99c79d244.txt",
          "passage": "¶ 1 While riding a roller coaster at the Washington State Fair, Jodi Brugh received a severe injury that resulted in a subdural hematoma that required brain surgery. Brugh sued Fun-Tastic Rides Co., Midway Rides LLC, and John Doe Manufacturer (collectively Fun-Tastic), alleging negligence. She relied on the doctrine of res ipsa loquitur to establish a breach of duty. Fun-Tastic moved for summary judgment on the theory that res ipsa loquitur did not apply, and after reconsideration, the trial cou",
          "source_url": null
        },
        {
          "evidence_id": "e5",
          "document_id": "passages/134df8b0193d148f5d975de8e0d84cf2d74aca80948fae45162c412cb3780246.txt",
          "citation": "passages/134df8b0193d148f5d975de8e0d84cf2d74aca80948fae45162c412cb3780246.txt",
          "passage": " create an inference of negligence where direct evidence is lacking. Defendants here have not cited to any cases where a party used res ipsa loquitur to defend itself against a claim of negligence or shift liability to a co-defendant. Nor is the Court aware of such authority. Accordingly, the Court finds that res ipsa loquitur does not apply.",
          "source_url": null
        }
      ],
      "limitations": [
        "No verified GPS coordinates; diagram is a chain-of-control schematic only.",
        "Graph and checklist summarize the generated answer and hypo; not independent legal advice."
      ],
      "fallback_text": "If Dotty asserts a claim against Canco based on negligence, the doctrine of *res ipsa loquitur* would likely not apply.\n\nThe doctrine of *res ipsa loquitur* permits negligence to be inferred from the physical cause of an injury, without the aid of circumstances pointing to the responsible human cause (Bowlin v. Duke Univ., 108 N.C.App. 145, 149, 423 S.E.2d 320, 322 (1992)). One of the criteria for applying *res ipsa loquitur* is that the event must be caused by an agency or instrumentality within the exclusive control of the defendant (St. Paul Fire & Marine Ins. Co. v. City of New York, 907 F.2d 299, 302 (2d Cir.1990)).\n\nIn this scenario, the facts state that the tuna consumed by Dotty and Guest came from a case that was knocked over by the plumbing contractor's workmen, causing the cans to be dented. The tuna in undamaged cans from the same Canco shipment was fit for consumption. This indicates that the instrumentality causing the tuna to become unfit for consumption (the denting of the cans) was the action of the plumbing contractor's workmen, not Canco. Since the damage occurred while the product was no longer in Canco's control, the requirement that the event be caused by an instrumentality within the exclusive control of the defendant (Canco) is not met.",
      "_source": {
        "query": "A water pipe burst in the basement of Supermart, a grocery store, flooding the basement and damaging cases of canned goods on the floor. The Plumbing contractor's workmen, in repairing the leak, knocked over several stacks of canned goods in cases, denting the cans. After settling its claims against the landlord for the water leak and against the plumbing contractor for the damage done by his workmen, Supermart put the goods on special sale. Four weeks later Dotty was shopping in Supermart. Several tables in the market were covered with assorted canned foods, all of which were dirty and dented. A sign on each of the tables read: \"Damaged Cans-Half Price.\" Dotty was having Guest for dinner that evening and purchased two dented cans of tuna, packed by Canco, from one of the tables displaying the damaged cans. Before Guest arrived, Dotty prepared a tuna casserole which she and Guest ate. Both became ill and the medical testimony established that the illness was caused by the tuna's being unfit for consumption. The tuna consumed by Dotty and Guest came from the case that was at the top of one of the stacks knocked over by the workmen. The tuna in undamaged cans from the same Canco shipment was fit for consumption.\nIf Dotty asserts a claim against Canco based on negligence, the doctrine of res ipsa loquitur is",
        "generated_answer": "If Dotty asserts a claim against Canco based on negligence, the doctrine of *res ipsa loquitur* would likely not apply.\n\nThe doctrine of *res ipsa loquitur* permits negligence to be inferred from the physical cause of an injury, without the aid of circumstances pointing to the responsible human cause (Bowlin v. Duke Univ., 108 N.C.App. 145, 149, 423 S.E.2d 320, 322 (1992)). One of the criteria for applying *res ipsa loquitur* is that the event must be caused by an agency or instrumentality within the exclusive control of the defendant (St. Paul Fire & Marine Ins. Co. v. City of New York, 907 F.2d 299, 302 (2d Cir.1990)).\n\nIn this scenario, the facts state that the tuna consumed by Dotty and Guest came from a case that was knocked over by the plumbing contractor's workmen, causing the cans to be dented. The tuna in undamaged cans from the same Canco shipment was fit for consumption. This indicates that the instrumentality causing the tuna to become unfit for consumption (the denting of the cans) was the action of the plumbing contractor's workmen, not Canco. Since the damage occurred while the product was no longer in Canco's control, the requirement that the event be caused by an instrumentality within the exclusive control of the defendant (Canco) is not met.",
        "model_id": "gemini-2.5-flash",
        "backend": "gemini"
      },
      "_story": {
        "takeaway": "Control over time — not a geographic map — explains why res ipsa fails against Canco.",
        "beats": []
      },
      "_story_timeline": {
        "items": []
      },
      "limited_evidence": false,
      "diagram_kind": "chain_of_control",
      "established_facts": [
        "Plumbing workmen knocked over stacks and dented cans at Supermart. (query)",
        "Undamaged cans from the same Canco shipment were fit for consumption. (query)",
        "Generated answer: res ipsa likely does not apply against Canco because exclusive control was broken."
      ],
      "not_established": [
        "Verified geographic coordinates for any location.",
        "That Canco retained exclusive control after the product left its shipment."
      ],
      "_audit": {
        "backed_items": 24,
        "unsupported_items": 0,
        "fabricated_numeric": 0,
        "limited_evidence": false,
        "primary": "map",
        "diagram_kind": "chain_of_control"
      },
      "additional_details": {
        "type": "matrix",
        "items": [
          {
            "id": "m1",
            "row": "Implies negligence",
            "column": "Supported?",
            "value": "Maybe",
            "symbol": "~",
            "description": "Illness after canned food — suggestive.",
            "evidence_ids": [
              "e1"
            ]
          },
          {
            "id": "m2",
            "row": "Exclusive control by Canco",
            "column": "Supported?",
            "value": "No",
            "symbol": "✗",
            "description": "Workmen intervene.",
            "evidence_ids": [
              "e1"
            ]
          },
          {
            "id": "m3",
            "row": "Plaintiff not at fault",
            "column": "Supported?",
            "value": "Likely",
            "symbol": "○",
            "description": "Consumer purchaser.",
            "evidence_ids": [
              "e1"
            ]
          },
          {
            "id": "m4",
            "row": "Res ipsa vs Canco",
            "column": "Supported?",
            "value": "Fails",
            "symbol": "✗",
            "description": "Generated answer.",
            "evidence_ids": [
              "e5"
            ]
          }
        ]
      }
    },
    {
      "query_id": "6",
      "query_type": "contractual",
      "title": "License rights granted — Corio / Changepoint",
      "summary": "Hierarchical breakdown of the exclusive hosting license: what rights Corio received, under what conditions, and what remains restricted.",
      "primary_view": {
        "type": "statute_tree",
        "items": [
          {
            "id": "root",
            "label": "Hosting Software\nLicense §2.1",
            "category": "grant",
            "detail": "Changepoint grants Corio a fee-bearing, irrevocable, nonexclusive, nontransferable license in the Territory.",
            "evidence_ids": [
              "e1"
            ]
          },
          {
            "id": "r1",
            "label": "Reproduce\nSoftware",
            "parent": "root",
            "category": "right",
            "detail": "Machine-executable object code only; installation on Corio Servers. No source-code access.",
            "evidence_ids": [
              "e1"
            ]
          },
          {
            "id": "r2",
            "label": "Host &\nAccess",
            "parent": "root",
            "category": "right",
            "detail": "Install multiple copies on Corio Servers; make remotely accessible to Customers for internal business purposes.",
            "evidence_ids": [
              "e2"
            ]
          },
          {
            "id": "r3",
            "label": "Sublicense\nto Customers",
            "parent": "root",
            "category": "right",
            "detail": "Sublicense an unlimited number of Customers to access and use the Software only through installation on Corio Servers.",
            "evidence_ids": [
              "e2"
            ]
          },
          {
            "id": "r4",
            "label": "Tools &\nUtilities",
            "parent": "root",
            "category": "right",
            "detail": "Use Changepoint's tools to configure, integrate, and manage the Software, subject to third-party restrictions.",
            "evidence_ids": [
              "e3"
            ]
          },
          {
            "id": "c1",
            "label": "Fee-bearing",
            "parent": "r1",
            "category": "condition",
            "detail": "License is contingent on payment of applicable fees as set forth in the Agreement.",
            "evidence_ids": [
              "e1"
            ]
          },
          {
            "id": "c2",
            "label": "Nonexclusive",
            "parent": "r1",
            "category": "condition",
            "detail": "Changepoint retains the right to grant identical or similar licenses to other parties.",
            "evidence_ids": [
              "e1"
            ]
          },
          {
            "id": "c3",
            "label": "Nontransferable\n(§14.1 exception)",
            "parent": "r1",
            "category": "restriction",
            "detail": "Corio may not assign or transfer the license except as expressly permitted under §14.1 (change of control).",
            "evidence_ids": [
              "e1",
              "e4"
            ]
          },
          {
            "id": "c4",
            "label": "Territory-limited",
            "parent": "r2",
            "category": "condition",
            "detail": "Rights are limited to the Territory as defined in the Agreement; cross-border deployment requires additional approval.",
            "evidence_ids": [
              "e1"
            ]
          },
          {
            "id": "c5",
            "label": "Internal Use\nOnly (Customers)",
            "parent": "r3",
            "category": "restriction",
            "detail": "Customers may access the Software only for their own internal business purposes; redistribution is prohibited.",
            "evidence_ids": [
              "e2"
            ]
          }
        ]
      },
      "secondary_views": [],
      "legend": [
        {
          "key": "grant",
          "label": "License grant",
          "color_token": "violet"
        },
        {
          "key": "right",
          "label": "Granted right",
          "color_token": "green"
        },
        {
          "key": "condition",
          "label": "Condition",
          "color_token": "amber"
        },
        {
          "key": "restriction",
          "label": "Restriction",
          "color_token": "red"
        }
      ],
      "annotations": [],
      "evidence_links": [
        {
          "evidence_id": "e1",
          "document_id": "cuad/CHANGEPOINTCORP_03_08_2000-EX-10.6-LICENSE AND HOSTING AGREEMENT.txt",
          "citation": "§2.1 Hosting Software License",
          "passage": "Changepoint hereby grants to Corio a fee-bearing, irrevocable, nonexclusive, nontransferable (except in accordance with Section 14.1 of this Agreement) license in the Territory to (i) reproduce the Software in machine executable object code format only for installation on the Corio Servers.",
          "source_url": null
        },
        {
          "evidence_id": "e2",
          "document_id": "cuad/CHANGEPOINTCORP_03_08_2000-EX-10.6-LICENSE AND HOSTING AGREEMENT.txt",
          "citation": "§2.1(ii)–(iv) Hosting Software License",
          "passage": "(ii) install multiple copies of the Software on Corio's Servers which will be made remotely accessible to Corio's Customers for their internal business purposes, (iii) permit limited access to and use of the Software by Customers through Corio Servers solely for such Customer's internal business purposes; (iv) sublicense an unlimited number of Customers to access and use the Software only through the installation on Corio Servers.",
          "source_url": null
        },
        {
          "evidence_id": "e3",
          "document_id": "cuad/CHANGEPOINTCORP_03_08_2000-EX-10.6-LICENSE AND HOSTING AGREEMENT.txt",
          "citation": "§2.1(v) Tools & Utilities",
          "passage": "(v) use Changepoint's tools and utilities, if any, subject to any restrictions placed on the Changepoint by third party software providers and payment of any applicable fees required by such third parties, to configure, integrate and manage the Software.",
          "source_url": null
        },
        {
          "evidence_id": "e4",
          "document_id": "cuad/CHANGEPOINTCORP_03_08_2000-EX-10.6-LICENSE AND HOSTING AGREEMENT.txt",
          "citation": "§2.4 Demonstration License",
          "passage": "Changepoint hereby grants to Corio a nonexclusive, nontransferable (except in accordance with Section 14.1 of this Agreement) royalty-free, fully paid up right and license in the Territory, on Corio Servers, to make a reasonable number of copies of the Demonstration Software solely for demonstration purposes to potential Customers.",
          "source_url": null
        }
      ],
      "limitations": [
        "License grant details apply to the 1999 Corio/Changepoint Agreement; subsequent amendments may alter scope.",
        "Redacted sections (CONFIDENTIAL TREATMENT) may contain fee schedules or territory definitions not shown here."
      ],
      "fallback_text": "Changepoint granted Corio a fee-bearing, irrevocable, nonexclusive, nontransferable hosting software license to reproduce, install, host, and sublicense the Software to Customers for their internal business purposes, subject to territory and fee conditions.",
      "_source": {
        "query": "Consider the License and Hosting Agreement between Corio Inc. and Changepoint, Inc.; What licenses are granted under this contract?",
        "generated_answer": "Under §2.1, Changepoint grants Corio a fee-bearing, irrevocable, nonexclusive, nontransferable license to: (i) reproduce the Software (object code only) for installation on Corio Servers; (ii) install multiple copies on Corio Servers remotely accessible to Customers; (iii) permit Customer access for internal business purposes; (iv) sublicense an unlimited number of Customers through Corio Servers; and (v) use Changepoint's tools and utilities. Additionally, §2.4 grants a separate royalty-free demonstration license.",
        "model_id": "gemini-2.5-flash",
        "backend": "gemini"
      },
      "limited_evidence": false,
      "_audit": {
        "backed_items": 10,
        "unsupported_items": 0,
        "fabricated_numeric": 0,
        "limited_evidence": false,
        "primary": "statute_tree",
        "diagram_kind": "rights hierarchy"
      }
    },
    {
      "query_id": "7",
      "query_type": "relational",
      "title": "NDA disclosure obligations — State of Illinois / Vendor",
      "summary": "Network of permitted and prohibited information flows under the Non-Disclosure Agreement: who the Vendor may or may not share Confidential Information with, and what exceptions apply.",
      "primary_view": {
        "type": "network",
        "items": [],
        "layout": "hierarchical",
        "network": {
          "nodes": [
            {
              "id": "state",
              "label": "State of Illinois\n(Discloser)",
              "entity_type": "party",
              "description": "Discloses fiber-optic network route information to Vendor under the NDA.",
              "evidence_ids": [
                "e1"
              ],
              "layer": 0
            },
            {
              "id": "ci",
              "label": "Confidential\nInformation",
              "entity_type": "doctrine",
              "description": "Specifications, designs, plans, drawings, and business/technical information proprietary to the State.",
              "evidence_ids": [
                "e1"
              ],
              "layer": 1
            },
            {
              "id": "vendor",
              "label": "Vendor",
              "entity_type": "defendant",
              "description": "Receives CI solely for discussing fiber-optic network data points and paths.",
              "evidence_ids": [
                "e1",
                "e2"
              ],
              "layer": 2
            },
            {
              "id": "employees",
              "label": "Need-to-Know\nEmployees",
              "entity_type": "actor",
              "description": "Vendor may share CI only with employees who have a demonstrated need to know and who are bound by the same obligations.",
              "evidence_ids": [
                "e2"
              ],
              "layer": 3
            },
            {
              "id": "third_parties",
              "label": "Third Parties\n(Prohibited)",
              "entity_type": "third_party",
              "description": "Vendor shall NOT disclose CI to any third party without the State's prior written approval.",
              "evidence_ids": [
                "e2"
              ],
              "layer": 3
            },
            {
              "id": "exceptions",
              "label": "Public Domain /\nPrior Knowledge",
              "entity_type": "actor",
              "description": "Restrictions do not apply if CI is independently developed, already public, or previously known to Vendor.",
              "evidence_ids": [
                "e3"
              ],
              "layer": 3
            }
          ],
          "edges": [
            {
              "source": "state",
              "target": "ci",
              "relation": "discloses",
              "description": "State shares proprietary fiber-optic network specifications as Confidential Information.",
              "evidence_ids": [
                "e1"
              ]
            },
            {
              "source": "ci",
              "target": "vendor",
              "relation": "received by",
              "description": "Vendor receives CI for a specific limited purpose (fiber project discussions).",
              "evidence_ids": [
                "e1"
              ]
            },
            {
              "source": "vendor",
              "target": "employees",
              "relation": "may share (need-to-know)",
              "description": "Permitted onward sharing only to employees with a need to know, who must be informed of NDA obligations.",
              "evidence_ids": [
                "e2"
              ]
            },
            {
              "source": "vendor",
              "target": "third_parties",
              "relation": "prohibited without approval",
              "description": "No disclosure to third parties unless the State grants prior written approval.",
              "evidence_ids": [
                "e2"
              ]
            },
            {
              "source": "ci",
              "target": "exceptions",
              "relation": "exceptions apply",
              "description": "Restrictions do not apply to independently developed, publicly available, or previously known information.",
              "evidence_ids": [
                "e3"
              ]
            }
          ]
        }
      },
      "secondary_views": [],
      "legend": [
        {
          "key": "party",
          "label": "Contracting party",
          "color_token": "teal"
        },
        {
          "key": "doctrine",
          "label": "Legal concept / CI",
          "color_token": "blue"
        },
        {
          "key": "actor",
          "label": "Permitted recipient",
          "color_token": "green"
        },
        {
          "key": "third_party",
          "label": "Prohibited recipient",
          "color_token": "orange"
        }
      ],
      "annotations": [],
      "evidence_links": [
        {
          "evidence_id": "e1",
          "document_id": "contractnli/DoiT-ICN-NonDisclosure-Agreement.txt",
          "citation": "NDA §1 — Purpose & Scope",
          "passage": "Vendor shall use such CONFIDENTIAL INFORMATION only for the purpose of discussing street level route information for the State's fiber optic network data points and paths with the State's support team in discussions regarding the State's fiber project.",
          "source_url": null
        },
        {
          "evidence_id": "e2",
          "document_id": "contractnli/DoiT-ICN-NonDisclosure-Agreement.txt",
          "citation": "NDA §1 — Disclosure Restrictions",
          "passage": "Vendor shall restrict disclosure of such CONFIDENTIAL INFORMATION to its employees with a need to know (and advise such employees of the obligations assumed herein), and shall not disclose such CONFIDENTIAL INFORMATION to any third party without prior written approval of the State.",
          "source_url": null
        },
        {
          "evidence_id": "e3",
          "document_id": "contractnli/DoiT-ICN-NonDisclosure-Agreement.txt",
          "citation": "NDA §1 — Exceptions",
          "passage": "These restrictions on the use or disclosure of CONFIDENTIAL INFORMATION shall not apply to any CONFIDENTIAL INFORMATION: i. which is independently developed by Vendor or is lawfully received free of restriction from another source having the right to so furnish such CONFIDENTIAL INFORMATION; ii. after it has become generally available to the public without breach of this Agreement by Vendor; iii. which, at the time of disclosure to Vendor, was known to Vendor to be free of restriction.",
          "source_url": null
        }
      ],
      "limitations": [
        "The NDA does not specify a duration for the confidentiality obligation; the three-year survival clause referenced in some DoiT agreements does not appear in this copy of the text.",
        "The State's identity is anonymized in the document header; factual background is drawn from the query."
      ],
      "fallback_text": "Under the NDA, the Vendor may share Confidential Information only with need-to-know employees and may not disclose it to any third party without prior written approval from the State. Exceptions apply if the information is independently developed, publicly available, or previously known to the Vendor.",
      "_source": {
        "query": "Consider the Non-Disclosure Agreement between DoiT and the State of Illinois; Who may the Vendor share Confidential Information with, and what third-party disclosures are prohibited?",
        "generated_answer": "The Vendor may share Confidential Information only with employees who have a need to know, after advising them of the NDA obligations. Disclosure to any third party requires the State's prior written approval. Restrictions do not apply to information that is independently developed, already public, or was already known to the Vendor at the time of disclosure.",
        "model_id": "gemini-2.5-flash",
        "backend": "gemini"
      },
      "limited_evidence": false,
      "_audit": {
        "backed_items": 14,
        "unsupported_items": 0,
        "fabricated_numeric": 0,
        "limited_evidence": false,
        "primary": "network",
        "diagram_kind": "obligation network"
      }
    },
    {
      "query_id": "8",
      "query_type": "analytical",
      "title": "Pre-closing conduct — Pandion / Merck merger",
      "summary": "Decision tree for whether Pandion Therapeutics may take a given action during the pre-closing period under the Agreement and Plan of Merger with Merck Sharp & Dohme Corp. (Feb 24, 2021).",
      "primary_view": {
        "type": "decision_tree",
        "items": [
          {
            "id": "root",
            "label": "Action planned during\npre-closing period?",
            "node_type": "question",
            "detail": "Section 7.1 governs all Company conduct between signing (Feb 24, 2021) and the Effective Time of the Merger.",
            "evidence_ids": [
              "e1"
            ]
          },
          {
            "id": "ordinary",
            "label": "Ordinary course of\nbusiness (past practice)",
            "parent": "root",
            "edge_label": "Routine / normal",
            "node_type": "question",
            "detail": "Company must conduct operations in all material respects consistent with past practice unless an exception applies.",
            "evidence_ids": [
              "e1"
            ]
          },
          {
            "id": "extraordinary",
            "label": "Material / extraordinary\naction",
            "parent": "root",
            "edge_label": "Out of ordinary course",
            "node_type": "question",
            "detail": "Material acquisitions, dispositions, equity issuances, debt incurrence, or other listed actions require Merck's written consent.",
            "evidence_ids": [
              "e2"
            ]
          },
          {
            "id": "allowed_ordinary",
            "label": "Proceed without\nMerck's consent",
            "parent": "ordinary",
            "edge_label": "Yes — consistent\nwith past practice",
            "node_type": "outcome",
            "outcome": "allowed",
            "detail": "Ordinary-course actions (hiring, routine R&D, vendor payments) may proceed without notifying Merck.",
            "evidence_ids": [
              "e1"
            ]
          },
          {
            "id": "need_consent",
            "label": "Seek Merck's prior\nwritten consent",
            "parent": "extraordinary",
            "edge_label": "Listed restricted\naction",
            "node_type": "question",
            "detail": "Examples include: acquiring a business worth ≥15% of consolidated assets, issuing new equity, or incurring material debt.",
            "evidence_ids": [
              "e2"
            ]
          },
          {
            "id": "consent_granted",
            "label": "Action permitted\n(consent received)",
            "parent": "need_consent",
            "edge_label": "Merck consents",
            "node_type": "outcome",
            "outcome": "allowed",
            "detail": "Merck's written consent removes the restriction for that specific action.",
            "evidence_ids": [
              "e3"
            ]
          },
          {
            "id": "consent_denied",
            "label": "Action prohibited\nuntil closing",
            "parent": "need_consent",
            "edge_label": "Merck declines\nor no response",
            "node_type": "outcome",
            "outcome": "prohibited",
            "detail": "Without consent, the action is prohibited during the pre-closing period; proceeding would constitute a breach.",
            "evidence_ids": [
              "e2"
            ]
          }
        ]
      },
      "secondary_views": [],
      "legend": [
        {
          "key": "question",
          "label": "Decision point",
          "color_token": "blue"
        },
        {
          "key": "allowed",
          "label": "Permitted outcome",
          "color_token": "green"
        },
        {
          "key": "prohibited",
          "label": "Prohibited outcome",
          "color_token": "red"
        }
      ],
      "annotations": [],
      "evidence_links": [
        {
          "evidence_id": "e1",
          "document_id": "maud/Pandion Therapeutics, Inc._Merck _ Co., Inc..txt",
          "citation": "§7.1(a) Conduct of the Company",
          "passage": "During the Pre-Closing Period, except (i) as set forth in Section 7.1 of the Company Disclosure Letter, (ii) as required by the express terms of this Agreement, or (iii) as otherwise consented to by Parent in writing, the Company shall, and shall cause each Company Subsidiary to, conduct its operations in all material respects in the ordinary course of business consistent with past practice.",
          "source_url": null
        },
        {
          "evidence_id": "e2",
          "document_id": "maud/Pandion Therapeutics, Inc._Merck _ Co., Inc..txt",
          "citation": "§7.1(b) Restricted Actions",
          "passage": "Without limiting the generality of the foregoing, during the Pre-Closing Period, except as expressly permitted or required by this Agreement or the Company Disclosure Letter, the Company shall not, and shall cause each Company Subsidiary not to, without the prior written consent of Parent: (i) issue, sell, or authorize any shares of capital stock; (ii) acquire or agree to acquire any assets constituting 15% or more of consolidated revenue, net income or assets; (iii) incur indebtedness for borrowed money in excess of applicable thresholds.",
          "source_url": null
        },
        {
          "evidence_id": "e3",
          "document_id": "maud/Pandion Therapeutics, Inc._Merck _ Co., Inc..txt",
          "citation": "§7.1 — Consent mechanism",
          "passage": "As used in this Agreement, 'ordinary course of business consistent with past practice' means, with respect to any action by the Company, an action that is consistent in nature, scope and magnitude with the past practices of the Company and that is taken in the ordinary course of the normal day-to-day operations of the Company.",
          "source_url": null
        }
      ],
      "limitations": [
        "The Company Disclosure Letter (§7.1 exceptions list) is not publicly available; additional permitted actions may exist.",
        "The merger closed in May 2021; this analysis reflects conditions as of the Feb 24, 2021 signing date.",
        "Thresholds for 'material' actions are defined by reference to consolidated financials not shown here."
      ],
      "fallback_text": "During the pre-closing period, Pandion must operate in the ordinary course of business. Any extraordinary or listed actions — such as issuing equity, making material acquisitions, or incurring significant debt — require Merck's prior written consent. Without consent, such actions are prohibited until the merger closes.",
      "_source": {
        "query": "Consider the Merger Agreement between Merck Sharp & Dohme Corp. and Pandion Therapeutics Inc.; What restrictions govern Pandion's conduct during the pre-closing period?",
        "generated_answer": "Under §7.1, Pandion must conduct its operations in the ordinary course of business consistent with past practice during the pre-closing period. It may not, without Merck's prior written consent, issue new equity, make material acquisitions (≥15% of consolidated assets), incur material debt, or take other listed extraordinary actions. Ordinary business operations may continue without seeking consent.",
        "model_id": "gemini-2.5-flash",
        "backend": "gemini"
      },
      "limited_evidence": false,
      "_audit": {
        "backed_items": 7,
        "unsupported_items": 0,
        "fabricated_numeric": 0,
        "limited_evidence": false,
        "primary": "decision_tree",
        "diagram_kind": "decision tree"
      }
    },
    {
      "query_id": "9",
      "query_type": "process",
      "title": "23andMe genetic data — sharing flows",
      "summary": "Flow diagram of how 23andMe collects, stores, and shares user genetic and personal information with research partners, service providers, and government-funded studies, depending on consent.",
      "primary_view": {
        "type": "map",
        "items": [
          {
            "id": "user",
            "label": "User / Sample",
            "entity_type": "plaintiff",
            "flow_order": 0,
            "detail": "User provides saliva sample and registration information (name, email, payment). Genetic information is generated from the sample.",
            "evidence_ids": [
              "e1"
            ]
          },
          {
            "id": "23andme",
            "label": "23andMe\nSystems",
            "entity_type": "store",
            "flow_order": 1,
            "detail": "23andMe collects, stores, and processes Registration Information, Genetic Information, Self-Reported Information, and Web Behavior Data.",
            "evidence_ids": [
              "e1",
              "e2"
            ]
          },
          {
            "id": "providers",
            "label": "Service\nProviders",
            "entity_type": "manufacturer",
            "flow_order": 2,
            "detail": "Third-party vendors acting on 23andMe's behalf for lab processing, customer support, IT, and payment. Bound by confidentiality.",
            "evidence_ids": [
              "e3"
            ]
          },
          {
            "id": "research",
            "label": "Research\nPartners",
            "entity_type": "retail",
            "flow_order": 2,
            "detail": "With user consent, Individual-level data is shared for 23andMe Research aimed at peer-reviewed publication.",
            "evidence_ids": [
              "e4"
            ]
          },
          {
            "id": "government",
            "label": "Government-\nFunded Studies",
            "entity_type": "location",
            "flow_order": 3,
            "detail": "Research funded by federal agencies (e.g., NIH). Participation requires explicit user consent. De-identified or aggregate data only.",
            "evidence_ids": [
              "e4"
            ]
          }
        ],
        "edges": [
          {
            "source": "user",
            "target": "23andme",
            "label": "Genetic + personal\ndata (registration)",
            "description": "User submits saliva sample and personal registration information; 23andMe generates genotype data.",
            "evidence_ids": [
              "e1"
            ]
          },
          {
            "source": "23andme",
            "target": "providers",
            "label": "Service delivery\n(no independent use)",
            "description": "Shared with vendors solely to perform services on 23andMe's behalf; vendors may not use data independently.",
            "evidence_ids": [
              "e3"
            ]
          },
          {
            "source": "23andme",
            "target": "research",
            "label": "With consent\n(individual-level)",
            "description": "Individual-level genetic and health data is shared with research partners only if user has given informed consent.",
            "evidence_ids": [
              "e4"
            ]
          },
          {
            "source": "research",
            "target": "government",
            "label": "Aggregate /\nde-identified only",
            "description": "Government-funded studies receive de-identified or aggregate data; individual re-identification is not permitted.",
            "evidence_ids": [
              "e4",
              "e5"
            ]
          }
        ]
      },
      "secondary_views": [],
      "legend": [
        {
          "key": "plaintiff",
          "label": "User / data subject",
          "color_token": "violet"
        },
        {
          "key": "store",
          "label": "23andMe platform",
          "color_token": "blue"
        },
        {
          "key": "manufacturer",
          "label": "Service providers",
          "color_token": "amber"
        },
        {
          "key": "retail",
          "label": "Research partners",
          "color_token": "teal"
        },
        {
          "key": "location",
          "label": "Government studies",
          "color_token": "gray"
        }
      ],
      "annotations": [],
      "evidence_links": [
        {
          "evidence_id": "e1",
          "document_id": "privacy_qa/23andMe.txt",
          "citation": "23andMe Privacy Statement — Data Types Collected",
          "passage": "23andMe collects and stores the following types of Personal Information: Registration Information: information you provide about yourself when registering for and/or purchasing our Services (e.g. name, email, address, user ID and password, and payment information). Genetic Information: information regarding your genotypes (i.e. the As, Ts, Cs, and Gs at particular locations in your genome), generated through processing of your saliva.",
          "source_url": null
        },
        {
          "evidence_id": "e2",
          "document_id": "privacy_qa/23andMe.txt",
          "citation": "23andMe Privacy Statement — Information Definitions",
          "passage": "Individual-level Information: information about a single individual's genotypes, diseases or other traits/characteristics, but which is not necessarily tied to Registration Information. Aggregate Information: information that has been combined with that of other users and analyzed or evaluated as a whole, such that no specific individual may be reasonably identified.",
          "source_url": null
        },
        {
          "evidence_id": "e3",
          "document_id": "privacy_qa/23andMe.txt",
          "citation": "23andMe Privacy Statement — Service Providers",
          "passage": "We use third-party service providers to perform services on our behalf, including laboratory processing, IT infrastructure, customer support, billing, and marketing. These service providers are bound by contractual obligations to keep personal information confidential and to use it only for the purposes for which we disclose it to them.",
          "source_url": null
        },
        {
          "evidence_id": "e4",
          "document_id": "privacy_qa/23andMe.txt",
          "citation": "23andMe Privacy Statement — Research Consent",
          "passage": "To allow you to share your Personal Information for 23andMe Research purposes. You have the choice to participate in 23andMe Research by providing your consent. '23andMe Research' refers to research aimed at publication in peer-reviewed journals and other research funded by the federal government (such as NIH).",
          "source_url": null
        },
        {
          "evidence_id": "e5",
          "document_id": "privacy_qa/23andMe.txt",
          "citation": "23andMe Privacy Statement — De-identification",
          "passage": "De-identified Information: information that has been stripped of your Registration Information (e.g., your name and contact information) and other identifying data such that you cannot reasonably be identified as an individual, also known as pseudonymized information.",
          "source_url": null
        }
      ],
      "limitations": [
        "Consent scope and revocation mechanisms are described in general terms; specific opt-in/opt-out workflows are not detailed in this excerpt.",
        "Insurers and employers: the privacy policy states that 23andMe will not share individual-level data with these parties without explicit consent, but this restriction is not shown in the flow above."
      ],
      "fallback_text": "23andMe collects genetic and personal information from users. With user consent, Individual-level data may be shared with research partners and government-funded studies. Without consent, only aggregate or de-identified data is shared. Service providers may access data solely to perform services on 23andMe's behalf.",
      "_source": {
        "query": "Consider '23andMe''s privacy policy; will my test results be shared with any third party entities?",
        "generated_answer": "Yes, but scope depends on consent. 23andMe shares data with service providers (for operational purposes) without additional consent. With your explicit consent, individual-level genetic and health data may be shared with research partners and federal-government-funded studies. Without consent, only aggregate or de-identified information is shared externally.",
        "model_id": "gemini-2.5-flash",
        "backend": "gemini"
      },
      "limited_evidence": false,
      "_audit": {
        "backed_items": 9,
        "unsupported_items": 0,
        "fabricated_numeric": 0,
        "limited_evidence": false,
        "primary": "map",
        "diagram_kind": "data flow"
      }
    },
    {
      "query_id": "10",
      "query_type": "comparative",
      "title": "Groupon data collection — purposes & safeguards",
      "summary": "Comparison of data types Groupon collects against commercial purposes and security controls. Each cell indicates whether that data type is used for the given purpose, as stated in the Groupon Privacy Statement.",
      "primary_view": {
        "type": "matrix",
        "items": [
          {
            "id": "m1",
            "row": "Personal Info\n(name, contact)",
            "column": "Personalization",
            "value": "Yes",
            "description": "Groupon uses personal info to customize site experience and service recommendations.",
            "evidence_ids": [
              "e2"
            ]
          },
          {
            "id": "m2",
            "row": "Personal Info\n(name, contact)",
            "column": "E-mail Marketing",
            "value": "Yes",
            "description": "Groupon uses contact info for promotional e-mails; users may opt out.",
            "evidence_ids": [
              "e2"
            ]
          },
          {
            "id": "m3",
            "row": "Personal Info\n(name, contact)",
            "column": "Partner Sharing",
            "value": "Permitted",
            "description": "Personal info may be shared with Business Partners for Other Programs; scope depends on enrollment.",
            "evidence_ids": [
              "e3"
            ]
          },
          {
            "id": "m4",
            "row": "Personal Info\n(name, contact)",
            "column": "Security Controls",
            "value": "Yes",
            "description": "Administrative, technical, and physical controls safeguard personal information.",
            "evidence_ids": [
              "e4"
            ]
          },
          {
            "id": "m5",
            "row": "Financial Info\n(payment data)",
            "column": "Personalization",
            "value": "No",
            "description": "Payment data is used for transaction processing, not for personalization.",
            "evidence_ids": [
              "e1"
            ]
          },
          {
            "id": "m6",
            "row": "Financial Info\n(payment data)",
            "column": "E-mail Marketing",
            "value": "No",
            "description": "Financial account information is not used for marketing communications.",
            "evidence_ids": [
              "e4"
            ]
          },
          {
            "id": "m7",
            "row": "Financial Info\n(payment data)",
            "column": "Partner Sharing",
            "value": "No",
            "description": "Financial account data is not shared with Business Partners.",
            "evidence_ids": [
              "e4"
            ]
          },
          {
            "id": "m8",
            "row": "Financial Info\n(payment data)",
            "column": "Security Controls",
            "value": "Yes",
            "description": "Industry-standard encryption is applied specifically to Financial Account Information.",
            "evidence_ids": [
              "e4"
            ]
          },
          {
            "id": "m9",
            "row": "Behavioral Data\n(browsing, purchases)",
            "column": "Personalization",
            "value": "Yes",
            "description": "Browsing and purchase history are used to recommend deals and personalize the site.",
            "evidence_ids": [
              "e2"
            ]
          },
          {
            "id": "m10",
            "row": "Behavioral Data\n(browsing, purchases)",
            "column": "E-mail Marketing",
            "value": "Yes",
            "description": "Past purchase behavior informs targeted promotional e-mails.",
            "evidence_ids": [
              "e2"
            ]
          },
          {
            "id": "m11",
            "row": "Behavioral Data\n(browsing, purchases)",
            "column": "Partner Sharing",
            "value": "Permitted",
            "description": "Aggregate behavioral data may be shared with advertising partners and vendors.",
            "evidence_ids": [
              "e3"
            ]
          },
          {
            "id": "m12",
            "row": "Behavioral Data\n(browsing, purchases)",
            "column": "Security Controls",
            "value": "Maybe",
            "description": "General information security program applies; no specific behavioral-data controls are described.",
            "evidence_ids": [
              "e4"
            ]
          }
        ]
      },
      "secondary_views": [],
      "legend": [
        {
          "key": "yes",
          "label": "Used for this purpose",
          "color_token": "green"
        },
        {
          "key": "no",
          "label": "Not used / not shared",
          "color_token": "red"
        },
        {
          "key": "maybe",
          "label": "Partial / unclear",
          "color_token": "amber"
        },
        {
          "key": "offered",
          "label": "Permitted (conditional)",
          "color_token": "blue"
        }
      ],
      "annotations": [],
      "evidence_links": [
        {
          "evidence_id": "e1",
          "document_id": "privacy_qa/Groupon.txt",
          "citation": "Groupon Privacy Statement — Data Collection",
          "passage": "We will collect information, including Personal Information and Non-Identifying Information, when you interact with us and the Site, for example when you: access or use the Site; register, subscribe, or create an account with Groupon; open or respond to our e-mails; purchase products or services on or through the Site.",
          "source_url": null
        },
        {
          "evidence_id": "e2",
          "document_id": "privacy_qa/Groupon.txt",
          "citation": "Groupon Privacy Statement — Data Use",
          "passage": "We (or our Vendors on our behalf), use information collected as described in this Privacy Statement to: Operate, maintain and improve the Site and our services; provide you with a personalized experience on our Site; send you promotional communications.",
          "source_url": null
        },
        {
          "evidence_id": "e3",
          "document_id": "privacy_qa/Groupon.txt",
          "citation": "Groupon Privacy Statement — Partner Sharing",
          "passage": "provide information to enroll or participate in Other Programs provided on behalf of, or together with, Business Partners; visit any page online that displays our ads or content.",
          "source_url": null
        },
        {
          "evidence_id": "e4",
          "document_id": "privacy_qa/Groupon.txt",
          "citation": "Groupon Privacy Statement — Security",
          "passage": "Groupon has implemented an information security program that contains administrative, technical and physical controls that are designed to reasonably safeguard Personal Information. For example, we use industry-standard encryption technology to secure Financial Account Information.",
          "source_url": null
        }
      ],
      "limitations": [
        "The 'Permitted' classification for partner sharing reflects the policy's broad language; actual sharing scope depends on user enrollment in 'Other Programs'.",
        "Groupon's cookie and tracking practices are described separately and are not captured in this matrix.",
        "Security controls are described qualitatively; no specific certifications (e.g., SOC 2, ISO 27001) are mentioned."
      ],
      "fallback_text": "Groupon collects personal information (name, contact), financial information (payment), and behavioral data (browsing, purchases). Personal and behavioral data are used for personalization, e-mail marketing, and may be shared with partners. Financial data receives the strongest security controls (encryption) but is not used for marketing. General security controls apply across all data types.",
      "_source": {
        "query": "Consider 'Groupon''s privacy policy; what does Groupon do with collected data, and what security controls are in place?",
        "generated_answer": "Groupon collects personal information, financial data, and behavioral data. It uses this information to operate and personalize its services, send promotional communications, and share with business partners for affiliated programs. Financial Account Information is specifically protected with industry-standard encryption. A broader information security program with administrative, technical, and physical controls applies to all Personal Information.",
        "model_id": "gemini-2.5-flash",
        "backend": "gemini"
      },
      "limited_evidence": false,
      "_audit": {
        "backed_items": 12,
        "unsupported_items": 0,
        "fabricated_numeric": 0,
        "limited_evidence": false,
        "primary": "matrix",
        "diagram_kind": "compliance matrix"
      }
    }
  ],
  "version": "20260812b"
};

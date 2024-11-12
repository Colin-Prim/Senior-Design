class BVHSkeleton:
    def __init__(self):
        self.joint_hierarchy = self.get_default_skeleton()

    def get_default_skeleton(self):
        return {
            "Hips": {
                "offset": [0.0, 0.0, 0.0],
                "channels": ["Xposition", "Yposition", "Zposition", "Xrotation", "Yrotation", "Zrotation"],
                "children": {
                    "Spine": {
                        "offset": [0.0, 10.0, 0.0],
                        "channels": ["Xrotation", "Yrotation", "Zrotation"],
                        "children": {
                            "Chest": {
                                "offset": [0.0, 10.0, 0.0],
                                "channels": ["Xrotation", "Yrotation", "Zrotation"],
                                "children": {
                                    "Neck": {
                                        "offset": [0.0, 5.0, 0.0],
                                        "channels": ["Xrotation", "Yrotation", "Zrotation"],
                                        "children": {
                                            "Head": {
                                                "offset": [0.0, 5.0, 0.0],
                                                "channels": ["Xrotation", "Yrotation", "Zrotation"],
                                                "children": {}
                                            }
                                        }
                                    },
                                    "LeftShoulder": {
                                        "offset": [5.0, 5.0, 0.0],
                                        "channels": ["Xrotation", "Yrotation", "Zrotation"],
                                        "children": {
                                            "LeftElbow": {
                                                "offset": [5.0, 0.0, 0.0],
                                                "channels": ["Xrotation", "Yrotation", "Zrotation"],
                                                "children": {
                                                    "LeftWrist": {
                                                        "offset": [5.0, 0.0, 0.0],
                                                        "channels": ["Xrotation", "Yrotation", "Zrotation"],
                                                        "children": {}
                                                    }
                                                }
                                            }
                                        }
                                    },
                                    "RightShoulder": {
                                        "offset": [-5.0, 5.0, 0.0],
                                        "channels": ["Xrotation", "Yrotation", "Zrotation"],
                                        "children": {
                                            "RightElbow": {
                                                "offset": [-5.0, 0.0, 0.0],
                                                "channels": ["Xrotation", "Yrotation", "Zrotation"],
                                                "children": {
                                                    "RightWrist": {
                                                        "offset": [-5.0, 0.0, 0.0],
                                                        "channels": ["Xrotation", "Yrotation", "Zrotation"],
                                                        "children": {}
                                                    }
                                                }
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    },
                    "LeftHip": {
                        "offset": [5.0, -10.0, 0.0],
                        "channels": ["Xrotation", "Yrotation", "Zrotation"],
                        "children": {
                            "LeftKnee": {
                                "offset": [0.0, -10.0, 0.0],
                                "channels": ["Xrotation", "Yrotation", "Zrotation"],
                                "children": {
                                    "LeftAnkle": {
                                        "offset": [0.0, -5.0, 0.0],
                                        "channels": ["Xrotation", "Yrotation", "Zrotation"],
                                        "children": {}
                                    }
                                }
                            }
                        }
                    },
                    "RightHip": {
                        "offset": [-5.0, -10.0, 0.0],
                        "channels": ["Xrotation", "Yrotation", "Zrotation"],
                        "children": {
                            "RightKnee": {
                                "offset": [0.0, -10.0, 0.0],
                                "channels": ["Xrotation", "Yrotation", "Zrotation"],
                                "children": {
                                    "RightAnkle": {
                                        "offset": [0.0, -5.0, 0.0],
                                        "channels": ["Xrotation", "Yrotation", "Zrotation"],
                                        "children": {}
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }


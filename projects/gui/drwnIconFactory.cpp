/*****************************************************************************
** DARWIN: A FRAMEWORK FOR MACHINE LEARNING RESEARCH AND DEVELOPMENT
** Distributed under the terms of the BSD license (see the LICENSE file)
** Copyright (c) 2007-2016, Stephen Gould
** All rights reserved.
**
******************************************************************************
** FILENAME:    drwnIconFactory.cpp
** AUTHOR(S):   Stephen Gould <stephen.gould@anu.edu.au>
**
*****************************************************************************/

#include <cstdlib>

#include "wx/wxprec.h"
#include "wx/utils.h"

#include "drwnBase.h"
#include "drwnIconFactory.h"

#include "resources/nodeGeneric.xpm"

#include "resources/nodeBinaryOp.xpm"
#include "resources/nodeBoosting.xpm"
#include "resources/nodeConcatenation.xpm"
#include "resources/nodeConfusionMatrix.xpm"
#include "resources/nodeDataExplorer.xpm"
#include "resources/nodeDecisionTree.xpm"
#include "resources/nodeDirSink.xpm"
#include "resources/nodeDirSource.xpm"
#include "resources/nodeDitherPlot.xpm"
#include "resources/nodeLinear.xpm"
#include "resources/nodeLogistic.xpm"
#include "resources/nodeLUTDecoder.xpm"
#include "resources/nodeLUTEncoder.xpm"
#include "resources/nodeMultiClassLDA.xpm"
#include "resources/nodePCA.xpm"
#include "resources/nodePRCurve.xpm"
#include "resources/nodeProjection.xpm"
#include "resources/nodeRescale.xpm"
#include "resources/nodeRollup.xpm"
#include "resources/nodeScatterPlot.xpm"
#include "resources/nodeTextFileSink.xpm"
#include "resources/nodeTextFileSource.xpm"
#include "resources/nodeUnaryOp.xpm"

#include "resources/nodeMatlab.xpm" // TODO: move this to plugin registry
#include "resources/nodeMATSink.xpm" // TODO: move this to plugin registry
#include "resources/nodeMATSource.xpm" // TODO: move this to plugin registry

#include "resources/nodeOpenCV.xpm" // TODO: move this to plugin registry
#include "resources/nodeOpenCVSink.xpm" // TODO: move this to plugin registry
#include "resources/nodeOpenCVSource.xpm" // TODO: move this to plugin registry

#include "resources/nodeLua.xpm" // TODO: move this to plugin registry
#include "resources/nodePython.xpm" // TODO: move this to plugin registry

// drwnIconFactory -----------------------------------------------------------

drwnIconFactory::drwnIconFactory() : _defaultIcon(NULL)
{
     // create default bitmap
    _defaultIcon = setIconMask(new wxBitmap(wxBITMAP(nodeGeneric)));

    // register standard types
    _registry["drwnBinaryOpNode"] = setIconMask(new wxBitmap(wxBITMAP(nodeBinaryOp)));
    _registry["drwnBoostedClassifierNode"] = setIconMask(new wxBitmap(wxBITMAP(nodeBoosting)));
    _registry["drwnConcatenationNode"] = setIconMask(new wxBitmap(wxBITMAP(nodeConcatenation)));
    _registry["drwnConfusionMatrixNode"] = setIconMask(new wxBitmap(wxBITMAP(nodeConfusionMatrix)));
    _registry["drwnDataExplorerNode"] = setIconMask(new wxBitmap(wxBITMAP(nodeDataExplorer)));
    _registry["drwnDecisionTreeNode"] = setIconMask(new wxBitmap(wxBITMAP(nodeDecisionTree)));
    _registry["drwnDitherPlotNode"] = setIconMask(new wxBitmap(wxBITMAP(nodeDitherPlot)));
    _registry["drwnExportFilesNode"] = setIconMask(new wxBitmap(wxBITMAP(nodeDirSink)));
    _registry["drwnImportFilesNode"] = setIconMask(new wxBitmap(wxBITMAP(nodeDirSource)));
    _registry["drwnLinearRegressionNode"] = setIconMask(new wxBitmap(wxBITMAP(nodeLinear)));
    _registry["drwnLinearTransformNode"] = setIconMask(new wxBitmap(wxBITMAP(nodeProjection)));
    _registry["drwnLUTDecoderNode"] = setIconMask(new wxBitmap(wxBITMAP(nodeLUTDecoder)));
    _registry["drwnLUTEncoderNode"] = setIconMask(new wxBitmap(wxBITMAP(nodeLUTEncoder)));
    _registry["drwnMultiClassLogisticNode"] = setIconMask(new wxBitmap(wxBITMAP(nodeLogistic)));
    _registry["drwnMultiClassLDANode"] = setIconMask(new wxBitmap(wxBITMAP(nodeMultiClassLDA)));
    _registry["drwnPCANode"] = setIconMask(new wxBitmap(wxBITMAP(nodePCA)));
    _registry["drwnPRCurveNode"] = setIconMask(new wxBitmap(wxBITMAP(nodePRCurve)));
    _registry["drwnRescaleNode"] = setIconMask(new wxBitmap(wxBITMAP(nodeRescale)));
    _registry["drwnRollupNode"] = setIconMask(new wxBitmap(wxBITMAP(nodeRollup)));
    _registry["drwnScatterPlotNode"] = setIconMask(new wxBitmap(wxBITMAP(nodeScatterPlot)));
    _registry["drwnTextFileSinkNode"] = setIconMask(new wxBitmap(wxBITMAP(nodeTextFileSink)));
    _registry["drwnTextFileSourceNode"] = setIconMask(new wxBitmap(wxBITMAP(nodeTextFileSource)));
    _registry["drwnUnaryOpNode"] = setIconMask(new wxBitmap(wxBITMAP(nodeUnaryOp)));

    // matlab plugin
    _registry["drwnMatlabNode"] = setIconMask(new wxBitmap(wxBITMAP(nodeMatlab))); // TODO: move
    _registry["drwnMATFileSinkNode"] = setIconMask(new wxBitmap(wxBITMAP(nodeMATSink))); // TODO: move
    _registry["drwnMATFileSourceNode"] = setIconMask(new wxBitmap(wxBITMAP(nodeMATSource))); // TODO: move

    // opencv plugin
    _registry["drwnOpenCVImageSinkNode"] = setIconMask(new wxBitmap(wxBITMAP(nodeOpenCVSink))); // TODO: move
    _registry["drwnOpenCVImageSourceNode"] = setIconMask(new wxBitmap(wxBITMAP(nodeOpenCVSource))); // TODO: move
    _registry["drwnOpenCVResizeNode"] = setIconMask(new wxBitmap(wxBITMAP(nodeOpenCV))); // TODO: move
    _registry["drwnOpenCVFilterBankNode"] = setIconMask(new wxBitmap(wxBITMAP(nodeOpenCV))); // TODO: move
    _registry["drwnOpenCVIntegralImageNode"] = setIconMask(new wxBitmap(wxBITMAP(nodeOpenCV))); // TODO: move
    _registry["drwnOpenCVNeighborhoodFeaturesNode"] = setIconMask(new wxBitmap(wxBITMAP(nodeOpenCV))); // TODO: move

    // lua plugin
    _registry["drwnLuaNode"] = setIconMask(new wxBitmap(wxBITMAP(nodeLua))); // TODO: move
    _registry["drwnLuaDummyNode"] = setIconMask(new wxBitmap(wxBITMAP(nodeLua))); // TODO: remove

    // python plugin
    _registry["drwnPythonNode"] = setIconMask(new wxBitmap(wxBITMAP(nodePython))); // TODO: move
    _registry["drwnPythonDummyNode"] = setIconMask(new wxBitmap(wxBITMAP(nodePython))); // TODO: remove
}

drwnIconFactory::~drwnIconFactory()
{
    for (map<string, wxBitmap *>::iterator it = _registry.begin();
         it != _registry.end(); it++) {
        delete it->second;
    }

    if (_defaultIcon != NULL)
        delete _defaultIcon;
}

drwnIconFactory& drwnIconFactory::get()
{
    static drwnIconFactory factory;
    return factory;
}

void drwnIconFactory::registerIcon(const char *name, const wxBitmap *bmp)
{
    DRWN_ASSERT(name != NULL);
    DRWN_ASSERT(bmp != NULL);

    map<string, wxBitmap *>::iterator it = _registry.find(string(name));
    if (it != _registry.end()) {
        DRWN_LOG_WARNING("replacing existing icon for \"" << name << "\"");
        delete it->second;
        it->second = setIconMask(new wxBitmap(*bmp));
    } else {
        _registry[string(name)] = setIconMask(new wxBitmap(*bmp));
    }
}

void drwnIconFactory::unregisterIcon(const char *name)
{
    DRWN_ASSERT(name != NULL);

    map<string, wxBitmap *>::iterator it = _registry.find(string(name));
    if (it != _registry.end()) {
        delete it->second;
        _registry.erase(it);
    } else {
        DRWN_LOG_ERROR("icon \"" << name << "\" is not in the icon registry");
    }
}

const wxBitmap *drwnIconFactory::getIcon(const char *name) const
{
    DRWN_ASSERT(name != NULL);
    map<string, wxBitmap *>::const_iterator it = _registry.find(string(name));
    if (it != _registry.end()) {
        DRWN_ASSERT(it->second != NULL);
        return it->second;
    }

    DRWN_ASSERT(_defaultIcon != NULL);
    return _defaultIcon;
}

wxBitmap *drwnIconFactory::setIconMask(wxBitmap *bmp)
{
    wxImage bmpImg = bmp->ConvertToImage();
    wxColour maskColour(bmpImg.GetRed(0,0), bmpImg.GetGreen(0,0), bmpImg.GetBlue(0,0));

    bmp->SetMask(new wxMask(*bmp, maskColour));
    return bmp;
}

